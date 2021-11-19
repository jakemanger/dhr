import numpy as np
from mctnet.utils import generate_kernel
from mctnet.data_loading import _load_point_data
from mctnet.utils import nn
import warnings

def apply_gaussian_kernel(array, indices, l, sigma):
    print(f'Creating {len(indices[0])} gaussian distributed points')
    
    for i in range(len(indices[0])):
        # generate kernel
        kernel = generate_kernel(l=l, sigma=sigma)

        # napari.view_image(kernel)
        # breakpoint()
        
        # find indices
        ind = np.array([indices[0][i], indices[1][i], indices[2][i]])
        min_ind = ind - l
        max_ind = ind + l

        # calculate indices of window (to cut out/slice)
        x_min = int(max(min_ind[0], 0))
        x_max = int(min(max_ind[0], array.shape[1] - 1))
        y_min = int(max(min_ind[1], 0))
        y_max = int(min(max_ind[1], array.shape[2] - 1))
        z_min = int(max(min_ind[2], 0))
        z_max = int(min(max_ind[2], array.shape[3] - 1))

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            warnings.warn(f'Point data at index {i} is outside the range of the volume/image (dims: {array.shape}). \
            The volume file may be mislabelled, have the wrong orientation or be corrupted.')

        else:
            # apply kernel to window

            # get old array values
            old_array_val = array[0, x_min:x_max+1, y_min:y_max+1, z_min:z_max+1].copy()

            # trim the kernel to size if needed
            if old_array_val.shape != kernel.shape:
                lower_lim_correction = np.zeros(3, dtype=int)
                upper_lim_correction = np.zeros(3, dtype=int)

                for ii in range(len(ind)):
                    lower_lim_correction[ii] = 0 if min_ind[ii] >= 0 else -min_ind[ii]
                    upper_lim_correction[ii] = 0 if max_ind[ii] < array.shape[ii + 1] else max_ind[ii] - array.shape[ii + 1] + 1

                kernel = kernel[
                    (lower_lim_correction[0]):(kernel.shape[0] - upper_lim_correction[0]),
                    (lower_lim_correction[1]):(kernel.shape[1] - upper_lim_correction[1]),
                    (lower_lim_correction[2]):(kernel.shape[2] - upper_lim_correction[2])
                ]

            if old_array_val.shape != kernel.shape:
                raise NotImplementedError('The shape of the kernel and the indexed image do not match. This should never happen.')

            # change kernel value only if it is greater than the previous value
            # so you can have locations that are very close together that won't
            # overlap and overwrite previously applied kernel values
            array[0, x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = np.maximum(old_array_val, kernel)
    
    return array


def _point_to_segmentation_vol(image, cornea_locations, rhabdom_locations):
    print('converting point data to segmentation volume...')
    # create zeros matrix the size of original data
    print('creating a images with zeros')
    corneas = np.zeros(image.shape)
    rhabdoms = np.zeros(image.shape)
    
    print('adding positions of corneas and rhabdoms with a gaussian kernel')

    sigma = 2
    l = 7

    print(f'Found {len(cornea_locations[:, 0])} cornea locations')
 
    corneas = apply_gaussian_kernel(
        corneas,
        (
            cornea_locations[:, 2],
            cornea_locations[:, 1],
            cornea_locations[:, 0]
        ),
        l,
        sigma
    )

    print(f'Found {len(rhabdom_locations[:, 0])} rhabdom locations')
    
    rhabdoms = apply_gaussian_kernel(
        rhabdoms,
        (
            rhabdom_locations[:, 2],
            rhabdom_locations[:, 1],
            rhabdom_locations[:, 0]
        ),
        l,
        sigma
    )

    return corneas, rhabdoms


def create_annotated_volumes(dir, image, swap_xy, resample_ratio):
    cornea_locations, rhabdom_locations = _load_point_data(dir, swap_xy)
    # apply the same resampling that was made to the image, so that 
    # annotated features line up correctly
    cornea_locations = np.rint(cornea_locations / resample_ratio)
    rhabdom_locations = np.rint(rhabdom_locations / resample_ratio)
    corneas, rhabdoms = _point_to_segmentation_vol(
        image,
        cornea_locations,
        rhabdom_locations
    )

    return corneas, rhabdoms