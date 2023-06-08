from skimage.measure import label
from scipy.ndimage.measurements import center_of_mass
from copy import deepcopy
import numpy as np


def check_bounds(x, y, z, min_vals, max_vals):
    if x > min_vals[0] and y > min_vals[1] and z > min_vals[2]:
        if x < max_vals[0] and y < max_vals[1] and z < max_vals[2]:
            return (x, y, z)
    return None


def locate_peaks_in_volume(volume, min_val, min_dist_from_border=0, relative=True):
    """Locates the maximum values in a volume.

    Locates peaks in heatmaps by finding their center of mass

    Args:
        volume (np.array): The volume to locate peaks in.
        min_val (float): The minimum value found in a peak (the threshold).
        min_dist_from_border (float): The minimum distance a point can be from the
        border.
        relative (bool): Whether the min_val should be calculated relative to the
        minimum and maximum voxel in the volume. Should be True if sampling with a
        point in the area 100% of the time.

    Returns:
        np.ndarray: The coordinates of the peaks in a 2D array.
    """
    if relative:
        min = np.min(volume)
        max = np.max(volume)
        min_val = (max - min) * min_val + min
    mask = deepcopy(volume)
    mask[volume >= min_val] = 1
    mask[volume < min_val] = 0
    mask = mask.squeeze()  # drop 4th dimension, as should only be of size 1

    label_regions, num_regions = label(mask, background=0, return_num=True)
    indexlist = [item for item in range(1, num_regions + 1)]
    centers_of_mass = center_of_mass(mask, label_regions, indexlist)

    # filter out points too close to the border
    mask_shape = mask.shape
    min_vals = (min_dist_from_border, min_dist_from_border, min_dist_from_border)
    max_vals = (
        mask_shape[0] - min_dist_from_border,
        mask_shape[1] - min_dist_from_border,
        mask_shape[2] - min_dist_from_border
    )
    test = [check_bounds(x, y, z, min_vals, max_vals) for x, y, z in centers_of_mass]
    test = [i for i in test if i is not None]

    return test
