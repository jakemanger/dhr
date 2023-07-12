import torchio as tio
import numpy as np
from typing import Tuple
import warnings


def resample_by_ratio(img: tio.Image, ratio: float) -> tio.Image:
    """Resamples a torchio Image `img` by a `ratio` from 0 to 1.

    Args:
        img (torchio.Image): Image to resample.
        ratio (float): Ratio to resample by.

    Returns:
        torchio.Image: Resampled image.
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
        subject (torchio.Subject): Subject to crop. Should contain an torchio.Image called 'img'.
        bbox (np.ndarray): Bounding box of the mask.
        mask (np.ndarray): Mask to crop.
        buffer (int): Buffer to crop by.

    Returns:
        torchio.Subject: Cropped subject. Contains an torchio.Image called 'img'.
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


def crop_using_bbox(
    subject: tio.Subject,
    bbox: np.ndarray,
) -> tio.Subject:
    """Crop a subject using a bounding box.

    Args:
        subject (torchio.Subject): Subject to crop. Should contain an torchio.Image
        called 'img'.
        bbox (np.ndarray): Bounding box of the mask.
    Returns:
        torchio.Subject: Cropped subject. Contains an torchio.Image called 'img'.
    """
    assert bbox[0].shape[0] == 3 and bbox[1].shape[0] == 3 \
        and len(bbox) == 2, 'bbox must be a 2d tuple of 3 mins and 3 maxes'

    mask = np.zeros(subject.img.shape)

    mask[
        0,
        bbox[0][0]:bbox[1][0],
        bbox[0][1]:bbox[1][1],
        bbox[0][2]:bbox[1][2]
    ] = 1

    # crop using mask
    subject = tio.Subject(
        img=subject.img,
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
    transformed.mask = None
    return transformed


def bbox_mask(mask_volume: np.ndarray, buffer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return 6 coordinates of a 3D bounding box from a given mask.

    Inspired by this SO question
    `<https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array>`.
    and a function in torchio.

    Args:
        mask_volume: 3D NumPy array.
        buffer: Buffer to add to the bounding box from the edges of the mask.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of minimum values ([0]) followed by maximum
        values ([1]).
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


def update_coords_after_crop(
    coords: np.ndarray,
    bbox: np.ndarray
) -> np.ndarray:
    """Update coordinates after a crop.

    Uses the bounding box (`bbox`) used to crop the image to update the coordinates.

    Args:
        coords (np.ndarray): Coordinates to update.
        bbox (np.ndarray): Bounding box of the mask.

    Returns:
        np.ndarray: Updated coordinates.
    """
    assert coords.shape[1] == 3, 'coords dimension at index 1 must be 3'
    assert bbox[0].shape[0] == 3 and bbox[1].shape[0] == 3 \
        and len(bbox) == 2, 'bbox must be a 2d tuple of 3 mins and 3 maxes'

    # remove any coordinates outside the maximum bounds of the bounding box
    indx = (coords[:, 0] < bbox[1][0]) & (coords[:, 1] < bbox[1][1]) & (coords[:, 2] < bbox[1][2])

    if any(indx == False):
        warnings.warn(
            'Some coordinates are outside the bounding box.'
            'Creation of the coordinates or the bounding box must have been incorrect.'
        )
        coords = coords[indx, :]

    # offset the coordinates to the minimum values of the new bounding box
    coords = coords - bbox[0]

    return coords


def crop_3d_coords(coords:np.ndarray, bounds:np.ndarray, correct_offset: bool = True) -> np.ndarray:
    """Crop 3d coordinates to min and max bounds.

    Args:
        coords (np.ndarray): Coordinates to crop.
        bounds (np.ndarray): Bounds to crop to. Structured as (xmin, ymin, zmin, xmax, ymax, zmax).
        correct_offset (bool): Whether to correct the offset of the coordinates, so they start at (0, 0, 0).
        If true, uses the minimum bounds of the bounding box to set (0, 0, 0).
    
    Returns:
        np.ndarray: Cropped coordinates.
    """
    assert coords.shape[1] == 3, 'coords second dimension must be 3.'
    assert len(bounds) == 6, 'bounds must be structured as (xmin, ymin, zmin, xmax, ymax, zmax).'
    within_min = (coords[:, 0] >= bounds[0]) & (coords[:, 1] >= bounds[1]) & (coords[:, 2] >= bounds[2])
    within_max = (coords[:, 0] < bounds[3]) & (coords[:, 1] < bounds[4]) & (coords[:, 2] < bounds[5])
    coords = coords[(within_min) & (within_max), :]

    if correct_offset:
        min = np.array([bounds[0], bounds[1], bounds[2]])
        coords = coords - min

    return coords
