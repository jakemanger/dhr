import torchio as tio
import numpy as np
from typing import Tuple

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

def crop_subject_with_mask_and_buffer(
    subject: tio.Subject,
    mask: np.ndarray,
    buffer: int
) -> tio.Subject:
    """Crop a subject with a mask and a buffer.

    Args:
        subject (torchio.Subject): Subject to crop.
        mask (np.ndarray): Mask to crop.
        buffer (int): Buffer to crop by.

    Returns:
        torchio.Subject: Cropped subject.
    """

    bbox = bbox_mask(mask.squeeze(), buffer)
    mask[
        0,
        bbox[0][0]:bbox[1][0],
        bbox[0][1]:bbox[1][1],
        bbox[0][2]:bbox[1][2]
    ] = 1

    # crop using mask
    subject = tio.Subject(
        img=subject.img,
        corneas=subject.corneas,
        rhabdoms=subject.rhabdoms,
        mask=tio.LabelMap(
            tensor=mask,
            affine=subject.img.affine,
            orientation=subject.img.orientation,
            spacing=subject.img.spacing,
        )
    )
    transform = tio.CropOrPad(
        mask_name='mask'
    )
    transformed = transform(subject)
    return transformed


def bbox_mask(mask_volume: np.ndarray, buffer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return 6 coordinates of a 3D bounding box from a given mask.

    Taken from `this SO question <https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array>`_.
    Modified from torchio.

    Args:
        mask_volume: 3D NumPy array.
        buffer: Buffer to add to the bounding box from the edges of the mask.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of minimum values ([0]) followed by maximum values ([1]).
    """
    min_vals = (0, 0, 0)
    max_vals = mask_volume.shape

    i_any = np.any(mask_volume, axis=(1, 2))
    j_any = np.any(mask_volume, axis=(0, 2))
    k_any = np.any(mask_volume, axis=(0, 1))
    i_min, i_max = np.where(i_any)[0][[0, -1]]
    j_min, j_max = np.where(j_any)[0][[0, -1]]
    k_min, k_max = np.where(k_any)[0][[0, -1]]
    bb_min = np.array([i_min, j_min, k_min]) - buffer
    bb_max = np.array([i_max, j_max, k_max]) + 1 + buffer

    bb_min = np.maximum(bb_min, min_vals)
    bb_max = np.minimum(bb_max, max_vals)
    return bb_min, bb_max