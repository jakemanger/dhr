import numpy as np
from dask_image.imread import imread
import scipy.io
import pandas as pb
from scipy.ndimage.morphology import distance_transform_edt
from skimage.util.shape import view_as_windows


def load_data(mct_file_path, annotation_file_path):
    print('loading data...')
    # load mct data
    stack = imread(mct_file_path)

    # load classifications
    mat = scipy.io.loadmat(annotation_file_path)
    classification = pb.DataFrame(
        mat['save_dat'][0]['data'][0][0][0][0]).iloc[:, 5]
    points = pb.DataFrame(
        mat['save_dat'][0]['data'][0][0][0][0]).iloc[:, [2, 0, 1]]
    points.columns = ['x', 'y', 'z']
    # convert back to numpy array and round to nearest pixel
    points = np.array(points)
    points = np.round(points).astype(int)
    # get corneas and rhabdom locations with x, y and z data
    cornea_indx = (classification == 0) | (classification == 2)
    rhabdom_indx = (classification == 1) | (classification == 3)
    cornea_locations = points[cornea_indx, :]
    rhabdom_locations = points[rhabdom_indx, :]

    return stack, cornea_locations, rhabdom_locations


def get_distance_prob_to_point(stack, cornea_locations, rhabdom_locations):
    print('getting distance probabilities for corneas and rhabdoms...')
    # create empty matrix the size of original data
    empty = stack.compute()
    empty[:, :, :] = 0
    np.unique(empty)

    corneas = empty.copy()
    rhabdoms = empty.copy()
    corneas[
        cornea_locations[:, 0],
        cornea_locations[:, 1],
        cornea_locations[:, 2]
    ] = 1
    rhabdoms[
        rhabdom_locations[:, 0],
        rhabdom_locations[:, 1],
        rhabdom_locations[:, 2]
    ] = 1
    
    # now calculate a distance matrix from our manually selected points
    cornea_dist = distance_transform_edt(1-corneas)
    rhabdom_dist = distance_transform_edt(1-rhabdoms)
    max_distance = 30
    cornea_prob = (max_distance - cornea_dist) / max_distance
    rhabdom_prob = (max_distance - rhabdom_dist) / max_distance
    cornea_prob[cornea_prob < 0] = 0
    rhabdom_prob[rhabdom_prob < 0] = 0

    return cornea_prob, rhabdom_prob


def convert_6d_to_4d(d):
    d_dims = d.shape
    if len(d_dims) != 6:
        raise Exception('convert_6d_to_4d() requires d to have 6 dimensions')
    # colapse to 4D array (list of 3d arrays)
    d = d.reshape(d_dims[0]*d_dims[1]*d_dims[2], d_dims[3], d_dims[4], d_dims[5])
    return d


def split_data_into_windows(d):
    print('splitting data into windows...')
    window_size = 31
    window_dims = (window_size, window_size, window_size)
    step_size = 5

    window_d = view_as_windows(d, window_dims, step_size)
    window_d = convert_6d_to_4d(window_d)
    return window_d
