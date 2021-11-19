import scipy.io
import pandas as pd
import numpy as np
import h5py

def _load_point_data(dir, swap_xy):
#    print('loading point data from .mat files...')
    # load classifications
    if h5py.is_hdf5(dir):
        f = h5py.File(dir, mode='r')
        classification = pd.DataFrame(np.array(f['save_dat']['data']['marked'])).iloc[5, :]
        file_type = f['save_dat']['stack']['image_format'][()].tobytes()[::2].decode()
#        print(f'Detected {file_type} file type')
        if swap_xy:
            points = pd.DataFrame(np.array(f['save_dat']['data']['marked'])).iloc[[2, 0, 1], :].T
        else:
            points = pd.DataFrame(np.array(f['save_dat']['data']['marked'])).iloc[[2, 1, 0], :].T
    else:
        mat = scipy.io.loadmat(dir)
        classification = pd.DataFrame(mat['save_dat'][0]['data'][0][0][0][0]).iloc[:,5]
        file_type = mat['save_dat'][0]['stack'][0]['image_format'][0][0][0]
#        print(f'Detected {file_type} file type')
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

    return cornea_locations, rhabdom_locations