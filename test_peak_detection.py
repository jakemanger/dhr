import time
import numpy as np
import torch
import napari
from deep_radiologist.heatmap_peaker import locate_peaks_in_volume, locate_peaks_in_volume_using_maximum_filter


def plot_heatmap_with_peaks(volume, center_of_mass_peaks, max_filter_peaks):
    # Plot the heatmap along with the located peaks using Napari
    with napari.gui_qt():
        viewer = napari.view_image(volume, name='Heatmap')
        viewer.add_points(center_of_mass_peaks, size=5, face_color='red', name='Center of mass peaks')
        viewer.add_points(max_filter_peaks, size=5, face_color='blue', name='Max filter peaks')


def create_test_volume_with_peaks(shape=(100, 100, 100), num_peaks=5, peak_std=5):
    # Create a zero-filled volume
    volume = np.zeros(shape)

    # Generate random coordinates for the peaks
    peak_coords = np.random.randint(0, shape[0], size=(num_peaks, 3))

    # Generate random peak intensities (simulating model that doesn't have too much confidence in prediction
    peak_intensities = np.random.random((num_peaks,))

    # Generate grid coordinates for the volume
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    # Add Gaussian peaks to the volume
    for i, coord in enumerate(peak_coords):
        x0, y0, z0 = coord
        volume += peak_intensities[i] * np.exp(-((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * peak_std ** 2))

    # Normalize the volume to have values between 0 and 1
    volume /= np.max(volume)

    # volume = np.expand_dims(volume, 0)

    return volume


if __name__ == '__main__':
    # Create a test volume with Gaussian peaks
    test_volume = create_test_volume_with_peaks(shape=(100, 100, 100), num_peaks=15)

    # Locate peaks in the test volume
    start_time = time.time()
    center_of_mass_peaks = locate_peaks_in_volume(test_volume, min_val=0.2, min_dist_from_border=5)
    end_time = time.time()
    print(f'Took {end_time - start_time} to complete center of mass peaks')

    start_time = time.time()
    max_filter_peaks = locate_peaks_in_volume_using_maximum_filter(test_volume, min_val=0.2, min_dist_from_border=5)
    end_time = time.time()
    print(f'Took {end_time - start_time} to complete using maximum filter')

    # Plot the heatmap along with the located peaks
    plot_heatmap_with_peaks(test_volume, center_of_mass_peaks, max_filter_peaks)
