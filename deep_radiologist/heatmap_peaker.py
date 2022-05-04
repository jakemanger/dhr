from skimage.feature import peak_local_max
import numpy as np


def locate_peaks_in_volume(volume, min_distance, min_val):
    """Locates the maximum values in a volume.

    Ignores peaks that are too close to each other or the edge of the volume, specified by min_distance.

    Args:
        volume (torch.Tensor): The volume to locate peaks in.
        min_distance (int): The minimum distance between peaks.
        min_val (float): The minimum value of a peak.

    Returns:
        np.ndarray: The coordinates of the peaks in a 2D array.
    """

    out = peak_local_max(volume.squeeze().astype(np.float32), min_distance=min_distance, threshold_abs=min_val, exclude_border=min_distance)

    return out
