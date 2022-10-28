from skimage.feature import peak_local_max
import numpy as np


def locate_peaks_in_volume(volume, min_distance, min_val):
    """Locates the maximum values in a volume.

    Ignores peaks that are too close to each other or the edge of the volume, specified by min_distance.

    Args:
        volume (torch.Tensor): The volume to locate peaks in.
        min_distance (int | str): The minimum distance between peaks.
            Can be a value (float) or a string specifying 'mean' or 'relative' (experimental methods).
            If 'mean', the mean value of the volume will be used or,
            if greater than the mean value, 0.1 will be used.
            If 'relative', the 0.5 * max value of the volume will be used or,
            if greater than the mean value, 0.05 will be used.

        min_val (float): The minimum value of a peak.

    Returns:
        np.ndarray: The coordinates of the peaks in a 2D array.
    """

    if min_val == 'mean':
        min_val = np.float32(max(np.mean(volume), 0.1))
    
    if min_val == 'relative':
        min_val = np.float32(max(np.max(volume) * 0.5, 0.05))

    out = peak_local_max(volume.squeeze().astype(np.float32), min_distance=min_distance, threshold_abs=min_val, exclude_border=min_distance)

    return out
