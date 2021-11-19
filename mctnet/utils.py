import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mctnet.data_loading import _load_point_data

class bcolors:
    """
    Just some helpful colours to use when printing to the console
    Example use:
    `print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")`
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def head_print(message):
    print(f'{bcolors.OKBLUE}{message}{bcolors.ENDC}')

def subhead_print(message):
    print(f'{bcolors.OKCYAN}{message}{bcolors.ENDC}')

def nn(x):
    """
    Calculates nearest neighbours, returns the distances and indices of the closest
    other value for each value
    """
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(x)
    distances, indices = nbrs.kneighbors(x)
    return distances, indices

def generate_kernel(l=5, sigma=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sigma`
    """
    
    x = np.arange(-l,l+1,1)   # coordinate arrays that must contain 0 (the center)
    y = x.copy()
    z = x.copy()
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))

    return kernel

def calculate_av_cornea_distance(dir, swap_xy=False):
    """
    Calcualtes the averae distance between corneas in units of voxels.
    If you want the units in mm, multiply the result by img.spacing
    """
    cornea_locations, _ = _load_point_data(dir, swap_xy)
    # work out average distance between corneas
    cornea_distances = nn(cornea_locations)
    av_cornea_distance = np.mean(pd.DataFrame(cornea_distances[0]).iloc[:, 1])

    return av_cornea_distance