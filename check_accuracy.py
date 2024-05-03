import numpy as np
from magicgui import magicgui
import torchio as tio
import argparse
import napari
import pathlib
import csv

points_classification = {} # Dictionary to keep track of point classifications

def load_nifti_image(file_path):
    img = tio.ScalarImage(
        file_path,
        check_nans=True
    )
    return img

def load_coords(file_path):
    locations = np.loadtxt(
        file_path,
        delimiter=',',
        ndmin=2,
        dtype=np.float64
    )
    return locations

@magicgui(call_button='Next Point')
def next_point(viewer: napari.Viewer, points_layer: napari.layers.Points):
    current_index = list(points_layer.selected_data)
    if not current_index:
        new_index = 0
    else:
        new_index = (current_index[0] + 1) % len(points_layer.data)
    points_layer.selected_data = {new_index}
    center_on_point(viewer, points_layer.data[new_index])

@magicgui(call_button='Previous Point')
def previous_point(viewer: napari.Viewer, points_layer: napari.layers.Points):
    current_index = list(points_layer.selected_data)
    if not current_index:
        new_index = 0
    else:
        new_index = (current_index[0] - 1) % len(points_layer.data)
    points_layer.selected_data = {new_index}
    center_on_point(viewer, points_layer.data[new_index])

def center_on_point(viewer, point):
    # Assuming ZYX order for point (modify if necessary)
    x, y, z = point
    print(f'centering to {x}, {y} and {z}')
    viewer.dims.set_point([0, 1, 2], [z, x, y])

def mark_as(points_layer: napari.layers.Points, classification: str, color: str):
    """Generic function to mark points with a given classification."""
    for index in points_layer.selected_data:
        points_classification[index] = classification
        points_layer.current_face_color = color
    points_layer.events.face_color()
    
@magicgui(call_button='Mark as TP')
def mark_as_tp(points_layer: napari.layers.Points):
    mark_as(points_layer, 'TP', 'green')

@magicgui(call_button='Mark as FP')
def mark_as_fp(points_layer: napari.layers.Points):
    mark_as(points_layer, 'FP', 'red')

@magicgui(call_button='Mark as FN')
def mark_as_fn(points_layer: napari.layers.Points):
    mark_as(points_layer, 'FN', 'yellow')

@magicgui(call_button='Mark as TN')
def mark_as_tn(points_layer: napari.layers.Points):
    mark_as(points_layer, 'TN', 'blue')


@magicgui(fn={'mode': 'w'}, call_button='Save Points')
def save_marked_points(points_layer: napari.layers.Points, viewer: napari.Viewer, fn=pathlib.Path.home()):
    fn = str(fn)
    if not fn.endswith('.csv'):
        fn += '.csv'

    with open(fn, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'z', 'classification']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for index, point in enumerate(points_layer.data):
            classification = points_classification.get(index, 'Unclassified')
            x, y, z = point
            writer.writerow({'x': x, 'y': y, 'z': z, 'classification': classification})

    print(f"Marked points saved to {fn}")


def main():
    parser = argparse.ArgumentParser(description='Napari tool for image and coords checking')
    parser.add_argument('image_path', type=str, help='Path to the NIfTI image file.')
    parser.add_argument('coords_path', type=str, help='Path to the CSV file containing coordinates.')
    args = parser.parse_args()

    image = load_nifti_image(args.image_path)

    transform = tio.ToCanonical()
    image = transform(image)

    coords = load_coords(args.coords_path)

    viewer = napari.Viewer()
    viewer.add_image(image.data.numpy(), name='NIfTI Image')
    points_layer = viewer.add_points(coords, size=5, face_color='pink', name='Coords')

    viewer.window.add_dock_widget(next_point, area='right')
    viewer.window.add_dock_widget(previous_point, area='right')
    viewer.window.add_dock_widget(mark_as_tp, area='right')
    viewer.window.add_dock_widget(mark_as_fp, area='right')
    viewer.window.add_dock_widget(mark_as_fn, area='right')
    viewer.window.add_dock_widget(mark_as_tn, area='right')
    viewer.window.add_dock_widget(save_marked_points, area='right')  # Add the save widget

    napari.run()

if __name__ == '__main__':
    main()
