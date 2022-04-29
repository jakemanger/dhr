import numpy as np
from deep_radiologist.utils import generate_kernel
from deep_radiologist.data_loading import _load_point_data
from deep_radiologist.utils import nn
import warnings


def apply_gaussian_kernel(array, indices, kernel, l):
    """ Apply gaussian values to an array at the given indices.

    Args:
        array (np.ndarray): Array to apply gaussian kernel to.
        indices (np.ndarray): Indices to apply gaussian kernel to.
        kernel (np.ndarray): Kernel to apply. Should be generated with `generate_kernel()`.
        l (int): Kernel size.

    Returns:
        np.ndarray: Array with gaussian kernel applied.
    """

    # print(f'Creating {len(indices[0])} gaussian distributed points')
    
    for i in range(len(indices[0])):
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
            old_array_val = array[0, x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

            # trim the kernel to size if needed
            if old_array_val.shape != kernel.shape:
                lower_lim_correction = np.zeros(3, dtype=int)
                upper_lim_correction = np.zeros(3, dtype=int)

                for ii in range(len(ind)):
                    lower_lim_correction[ii] = 0 if min_ind[ii] >= 0 else -min_ind[ii]
                    upper_lim_correction[ii] = 0 if max_ind[ii] < array.shape[ii + 1] else max_ind[ii] - array.shape[ii + 1] + 1

                trimmed_kernel = kernel.copy() # only copy the kernel if it needs to be trimmed to save computation time
                trimmed_kernel = trimmed_kernel[
                    (lower_lim_correction[0]):(trimmed_kernel.shape[0] - upper_lim_correction[0]),
                    (lower_lim_correction[1]):(trimmed_kernel.shape[1] - upper_lim_correction[1]),
                    (lower_lim_correction[2]):(trimmed_kernel.shape[2] - upper_lim_correction[2])
                ]
                this_kernel = trimmed_kernel
            else:
                this_kernel = kernel


            if old_array_val.shape != this_kernel.shape:
                raise NotImplementedError(
                    f'The shape of the kernel (this_kernel: {this_kernel.shape}) and the indexed image ({old_array_val.shape})'
                    ' do not match. This should never happen.'
                )

            # change kernel value only if it is greater than the previous value
            # so you can have locations that are very close together that won't
            # overlap and overwrite previously applied kernel values
            array[0, x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = np.maximum(old_array_val, this_kernel)
    
    return array


def _point_to_segmentation_vol(image, cornea_locations, rhabdom_locations):
    """ Create a segmentation volume from point data.

    Args:
        image (np.ndarray): Image to apply point data to.
        cornea_locations (np.ndarray): Locations of cornea points.
        rhabdom_locations (np.ndarray): Locations of rhabdom points.
            
        Returns:
            np.ndarray: Segmentation volume.
    """


    print('converting point data to segmentation volume...')
    # create zeros matrix the size of original data
    print('creating a images with zeros')
    corneas = np.zeros(image.shape)
    rhabdoms = np.zeros(image.shape)
    
    print('adding positions of corneas and rhabdoms with a gaussian kernel')


    print(f'Found {len(cornea_locations[:, 0])} cornea locations')

    # generate kernel
    l=7,
    sigma=2
    kernel = generate_kernel(l=l, sigma=sigma)

    corneas = apply_gaussian_kernel(
        corneas,
        (
            cornea_locations[:, 2],
            cornea_locations[:, 1],
            cornea_locations[:, 0]
        ),
        kernel,
        l
    )

    print(f'Found {len(rhabdom_locations[:, 0])} rhabdom locations')
    
    rhabdoms = apply_gaussian_kernel(
        rhabdoms,
        (
            rhabdom_locations[:, 2],
            rhabdom_locations[:, 1],
            rhabdom_locations[:, 0]
        ),
        kernel,
        l
    )

    return corneas, rhabdoms


def create_annotated_volumes(path, image, swap_xy, resample_ratio):
    """ Create heatmap annotated volumes from previous annotated locations at a path from mctv
    
    Uses corneas and rhabdom location data and an image with the same dimensions.

    Args:
        path (str): path to load point data from.
        image (str): Image with dimensions to apply point data to.
        swap_xy (bool): Whether to swap x and y axis.
        resample_ratio (float): Resample ratio. Use this if the image argument
        supplied was resampled after annotating.

    Returns:
        np.ndarray: Annotated volume with corneas
        np.ndarray: Annotated volume with rhabdoms
    """


    cornea_locations, rhabdom_locations = _load_point_data(path, swap_xy)
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
