import numpy as np
import pandas as pd
import napari
from magicgui import magicgui
import sys

def load_coordinates(file_path):
    """Load coordinates from a CSV file without headers."""
    df = pd.read_csv(file_path, header=None)
    return df.values  # Return all values as a numpy array

def save_coordinates(file_path, data):
    """Save coordinates to a CSV file without headers."""
    df = pd.DataFrame(data)
    df.to_csv(file_path, header=False, index=False)

@magicgui(call_button='Load Coordinates')
def load_button(file_path: str, viewer: napari.Viewer = None):
    """Load coordinates and add them to the viewer."""
    coordinates = load_coordinates(file_path)
    viewer.add_points(coordinates, size=5, face_color='cyan', name='Coordinates')

@magicgui(call_button='Save Coordinates')
def save_button(save_path: str, viewer: napari.Viewer = None):
    """Save edited coordinates to a specified path."""
    if viewer.layers and 'Coordinates' in viewer.layers:
        data = viewer.layers['Coordinates'].data
        save_coordinates(save_path, data)
        print(f"Coordinates saved to: {save_path}")
    else:
        print("No coordinates to save.")

def main(file_path):
    """Main function to run the Napari viewer with editing capabilities."""
    viewer = napari.Viewer()

    # Load initial coordinates
    load_button.file_path.value = file_path  # Pre-fill the load button's file path
    load_button(file_path=file_path)  # Load coordinates automatically on start

    # Set the save path to the original file path by default
    save_button.save_path.value = file_path  # Pre-fill the save path

    # Add the load and save buttons to the viewer
    viewer.window.add_dock_widget(load_button)
    viewer.window.add_dock_widget(save_button)

    # Start the Napari event loop
    napari.run()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)

