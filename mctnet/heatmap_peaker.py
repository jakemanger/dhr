from skimage.feature import peak_local_max

def locate_peaks_in_volume(volume, min_distance, min_val):
    """Locates the maximum values in a volume.

    Args:
        volume (torch.Tensor): The volume to locate peaks in.
        min_distance (int): The minimum distance between peaks.
        min_val (float): The minimum value of a peak.

    Returns:
        np.ndarray: The coordinates of the peaks in a 2D array.
    """

    out = peak_local_max(volume.squeeze(), min_distance=min_distance, threshold_abs=min_val)

    return out