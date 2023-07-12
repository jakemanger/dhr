import scipy.io
import pandas as pd
import numpy as np
import h5py
from pathlib import Path, PurePath


def _mat_to_np(path, swap_xy, label_name):
    """ Convert .mat files (from Jan Hemmi's mctv_mat software) to a numpy array """
    if h5py.is_hdf5(path):
        f = h5py.File(path, mode='r')
        classification = pd.DataFrame(
            np.array(f['save_dat']['data']['marked'])
        ).iloc[5, :]
        file_type = f['save_dat']['stack']['image_format'][()].tobytes()[::2].decode()
        print(f'Detected {file_type} file type')
        if swap_xy:
            points = pd.DataFrame(
                np.array(f['save_dat']['data']['marked'])
            ).iloc[[2, 0, 1], :].T
        else:
            points = pd.DataFrame(
                np.array(f['save_dat']['data']['marked'])
            ).iloc[[2, 1, 0], :].T
    else:
        mat = scipy.io.loadmat(path)
        classification = pd.DataFrame(
            mat['save_dat'][0]['data'][0][0][0][0]
        ).iloc[:,5]
        file_type = mat['save_dat'][0]['stack'][0]['image_format'][0][0][0]
        print(f'Detected {file_type} file type')
        if swap_xy:
            points = pd.DataFrame(mat['save_dat'][0]['data'][0][0][0][0]).iloc[:,[2, 0, 1]]
        else:
            points = pd.DataFrame(mat['save_dat'][0]['data'][0][0][0][0]).iloc[:,[2, 1, 0]]

    points.columns = ['x', 'y', 'z']

    # convert back to numpy array and round to nearest voxel
    points = np.array(points)
    points = np.round(points).astype(int)

    # get corneas and rhabdom locations with x, y and z data
    cornea_indx = (classification == 0) | (classification == 2)
    rhabdom_indx = (classification == 1) | (classification == 3)
    cornea_locations = points[cornea_indx, :]
    rhabdom_locations = points[rhabdom_indx, :]

    cornea_locations = np.array([
        cornea_locations[:, 2],
        cornea_locations[:, 1],
        cornea_locations[:, 0]
    ], dtype=float)

    rhabdom_locations = np.array([
        rhabdom_locations[:, 2],
        rhabdom_locations[:, 1],
        rhabdom_locations[:, 0]
    ], dtype=float)

    if label_name == 'corneas':
        return cornea_locations
    if label_name == 'rhabdoms':
        return rhabdom_locations

    raise NotImplementedError(
        'Only corneas and rhabdoms are implemented label name '
        'types in mctv'
    )


def mat_to_csv(path, new_path, swap_xy, label_name='corneas'):
    """ Convert .mat files (from Jan Hemmi's mctv_mat software) to a csv file """
    arr = _mat_to_np(path, swap_xy, label_name).T
    np.savetxt(new_path, arr, delimiter=",")
    print(f'Saved to {new_path}')
    return new_path


def _load_point_data(path, swap_xy, file_type='mctv_mat', label_name='corneas'):
    """ Load labels from a file.

    Args:
        path (str): Path to load point data from.
        file_type (str): The file type with your labels.
            Can be 'csv', or 'mctv_mat' (a software built by Jan Hemmi at UWA)

    Returns:
        np.ndarray: Locations of label_name.
    """
#    print('loading point data from .mat files...')
    # load classifications

    if file_type == 'csv':
        return pd.read_csv(path, sep=',').values
    elif file_type == 'mctv_mat':
        return _mat_to_np(path, swap_xy, label_name)
    else:
        raise NotImplementedError(
            f'file_type {file_type} has not yet been implemented. Raise an issue on'
            'the github repo or make a pull request, if you would like to add a new '
            'file type.'
        )
