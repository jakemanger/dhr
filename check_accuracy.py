import numpy as np
from magicgui import magicgui
import torchio as tio
import argparse
import napari
import pathlib
import csv
import nibabel as nib
import pandas as pd
import os
import pickle
import yaml
from yaml.loader import SafeLoader
import glob

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

def plot_and_edit_points(data):
    """Loads the mct (multi-channel test image) and allows interactive point editing in Napari."""
    mct = load_nifti_image(data['mct_path'])

    file_name_without_extension = os.path.splitext(os.path.basename(data['y_hat_path']))[0]
    
    print(f"Editing points for: {file_name_without_extension}")
    print("Instructions:")
    print("- Move points between layers to correct classifications")
    print("- Green = True Positives, Red = False Positives, Blue = False Negatives")
    print("- Press 's' to save when finished")
    print("- Close the viewer window when done")
    
    with napari.gui_qt():
        viewer = napari.Viewer()
        # Add the 3D image to the viewer
        viewer.add_image(mct, name='MCT Image')

        # Add true positives as points in green
        tps_layer = viewer.add_points(data['tps'], size=5, edge_color='green', face_color='green', name='True Positives')

        # Add false positives as points in red
        fps_layer = viewer.add_points(data['fps'], size=5, edge_color='red', face_color='red', name='False Positives')

        # Add false negatives as points in blue
        fns_layer = viewer.add_points(data['fns'], size=5, edge_color='blue', face_color='blue', name='False Negatives')

        # Enable editing for the points (tps, fps, fns)
        tps_layer.editable = True
        fps_layer.editable = True
        fns_layer.editable = True

        @viewer.bind_key('s')
        def save_points(viewer):
            """Save edited points when 's' is pressed."""
            # Create output directory if it doesn't exist
            os.makedirs('corrected_results', exist_ok=True)
            
            np.savetxt(f'corrected_results/{file_name_without_extension}_cleaned_tps.csv', 
                      tps_layer.data, delimiter=',', header='x,y,z', comments='')
            np.savetxt(f'corrected_results/{file_name_without_extension}_cleaned_fps.csv', 
                      fps_layer.data, delimiter=',', header='x,y,z', comments='')
            np.savetxt(f'corrected_results/{file_name_without_extension}_cleaned_fns.csv', 
                      fns_layer.data, delimiter=',', header='x,y,z', comments='')
            print(f"Edited points saved to corrected_results/ directory")

def load_cleaned_points(file_name_without_extension):
    """Load the cleaned points from CSV files."""
    try:
        tps_path = f'corrected_results/{file_name_without_extension}_cleaned_tps.csv'
        fps_path = f'corrected_results/{file_name_without_extension}_cleaned_fps.csv'
        fns_path = f'corrected_results/{file_name_without_extension}_cleaned_fns.csv'
        
        # Load cleaned points, handling empty files
        tps = np.loadtxt(tps_path, delimiter=',', skiprows=1, ndmin=2) if os.path.getsize(tps_path) > 0 else np.array([]).reshape(0, 3)
        fps = np.loadtxt(fps_path, delimiter=',', skiprows=1, ndmin=2) if os.path.getsize(fps_path) > 0 else np.array([]).reshape(0, 3)
        fns = np.loadtxt(fns_path, delimiter=',', skiprows=1, ndmin=2) if os.path.getsize(fns_path) > 0 else np.array([]).reshape(0, 3)
        
        return tps.tolist(), fps.tolist(), fns.tolist()
    except Exception as e:
        print(f"Error loading cleaned points for {file_name_without_extension}: {e}")
        return None, None, None

def recalculate_metrics(tps, fps, fns):
    """Recalculate performance metrics from corrected points."""
    num_tps = len(tps) if tps is not None else 0
    num_fps = len(fps) if fps is not None else 0
    num_fns = len(fns) if fns is not None else 0
    
    # Calculate precision, recall, and F1 score
    precision = num_tps / (num_tps + num_fps) if (num_tps + num_fps) > 0 else 0
    recall = num_tps / (num_tps + num_fns) if (num_tps + num_fns) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'num_tps_corrected': num_tps,
        'num_fps_corrected': num_fps,
        'num_fns_corrected': num_fns,
        'precision_corrected': precision,
        'recall_corrected': recall,
        'f1_corrected': f1_score
    }

def update_results_with_corrections(original_data):
    """Update the original data with corrected metrics."""
    file_name_without_extension = os.path.splitext(os.path.basename(original_data['y_hat_path']))[0]
    
    # Load corrected points
    tps_corrected, fps_corrected, fns_corrected = load_cleaned_points(file_name_without_extension)
    
    if tps_corrected is not None:
        # Recalculate metrics
        corrected_metrics = recalculate_metrics(tps_corrected, fps_corrected, fns_corrected)
        
        # Update the original data with corrected metrics
        original_data.update(corrected_metrics)
        original_data['tps_corrected'] = tps_corrected
        original_data['fps_corrected'] = fps_corrected
        original_data['fns_corrected'] = fns_corrected
        
        print(f"Updated metrics for {file_name_without_extension}:")
        print(f"  Original - TPs: {original_data['num_tps']}, FPs: {original_data['num_fps']}, FNs: {original_data['num_fns']}")
        print(f"  Corrected - TPs: {corrected_metrics['num_tps_corrected']}, FPs: {corrected_metrics['num_fps_corrected']}, FNs: {corrected_metrics['num_fns_corrected']}")
        print(f"  Original F1: {original_data.get('f1', 'N/A'):.3f}")
        print(f"  Corrected F1: {corrected_metrics['f1_corrected']:.3f}")
        print()
    else:
        print(f"No corrected data found for {file_name_without_extension}")
    
    return original_data

def main():
    # Load the pickle file with the best fiddlercrab cornea results
    # You'll need to adjust this path to match your specific pickle file
    pickle_files = glob.glob('*fiddlercrab_corneas*results.pickle')
    
    if not pickle_files:
        print("No fiddlercrab corneas pickle files found. Please ensure the pickle file is in the current directory.")
        return
    
    pickle_file = pickle_files[0]  # Use the first one found
    print(f"Loading data from: {pickle_file}")
    
    with open(pickle_file, 'rb') as f:
        data_list = pickle.load(f)
    
    print(f"Found {len(data_list)} items to review")
    
    # Process each item
    for i, data in enumerate(data_list):
        print(f"\nProcessing item {i+1}/{len(data_list)}")
        
        # Check if corrections already exist
        file_name_without_extension = os.path.splitext(os.path.basename(data['y_hat_path']))[0]
        corrected_files_exist = all(os.path.exists(f'corrected_results/{file_name_without_extension}_cleaned_{suffix}.csv') 
                                   for suffix in ['tps', 'fps', 'fns'])
        
        if not corrected_files_exist:
            print(f"Opening napari for manual correction...")
            plot_and_edit_points(data)
            print("Napari session completed. Processing corrections...")
        else:
            print(f"Corrected files already exist for {file_name_without_extension}")
        
        # Update with corrections
        data_list[i] = update_results_with_corrections(data)
    
    # Save the updated results
    corrected_pickle_file = pickle_file.replace('.pickle', '_corrected.pickle')
    with open(corrected_pickle_file, 'wb') as f:
        pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Updated results saved to: {corrected_pickle_file}")
    
    # Create a summary CSV of the corrections
    summary_data = []
    for data in data_list:
        if 'num_tps_corrected' in data:
            summary_data.append({
                'file': os.path.basename(data['y_hat_path']),
                'original_tps': data['num_tps'],
                'original_fps': data['num_fps'],
                'original_fns': data['num_fns'],
                'original_f1': data.get('f1', None),
                'corrected_tps': data['num_tps_corrected'],
                'corrected_fps': data['num_fps_corrected'],
                'corrected_fns': data['num_fns_corrected'],
                'corrected_f1': data['f1_corrected'],
                'f1_improvement': data['f1_corrected'] - (data.get('f1', 0) or 0)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('correction_summary.csv', index=False)
        print("\nSummary of corrections saved to: correction_summary.csv")
        print(summary_df)

if __name__ == "__main__":
    main()
