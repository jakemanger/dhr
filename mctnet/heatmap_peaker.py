
def locate_peaks_in_volume(volume, patch_size, stride):
    """Locates the maximum values in a volume.

    Args:
        volume (torch.Tensor): The volume to locate peaks in.
        patch_size (int): The size of the patch to use.
        stride (int): The stride to use.

    Returns:
        list: A list of tuples containing the coordinates of the peaks.
    """

    breakpoint()

    return []
    #return [(x, y, z)]



def _locate_peak_in_patch(patch):
    """Locates the maximum value in a patch.

    Args:
        patch (torch.Tensor): The patch to locate the peak in.

    Returns:
        tuple: A tuple containing the coordinates of the peak.
    """

    breakpoint()

    return 0
    #return index