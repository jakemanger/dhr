import torchio as tio

def resample_by_ratio(img, ratio):
    """Resamples a torchio ScalarImage `img` by a `ratio` from 0 to 1.

    Args:
        img (torchio.ScalarImage): Image to resample.
        ratio (float): Ratio to resample by.

    Returns:
        torchio.ScalarImage: Resampled image.
    """
    print(f'Image old spacing {img.spacing}')
    x, y, z = ratio * img.spacing[0], ratio * img.spacing[1], ratio * img.spacing[2]
    transform = tio.Resample((x, y, z))
    img = transform(img)
    print(f'Image new spacing {img.spacing}')
    return img
