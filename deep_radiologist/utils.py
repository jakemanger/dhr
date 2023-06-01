import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


class bcolors:
    """Just some helpful colours to use when printing to the console

    Example use:
        ```
        print(
            f"{bcolors.WARNING}Warning: No active frommets remain. "
            f"Continue?{bcolors.ENDC}"
        )
        ```
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
    """ Calculates nearest neighbours, returns the distances and indices of the closest
    other value for each value

    Args:
        x (np.array): 3-dimensional array of values to find nearest neighbours for

    Returns:
        tuple: distances and indices of the closest other value for each value
    """
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(x)
    distances, indices = nbrs.kneighbors(x)
    return distances, indices


def draw_voxels(array, x, y, z, value: float = 1., voxel_size: int = 1):
    """ Draws voxels at the specified 3D coordinates in the given numpy array.

    If the voxel_size is 1, the function uses trilinear interpolation to distribute
    the voxel's value among the eight surrounding cells of the array, for greater than
    voxel precision. This enables the voxel to be centered in between multiple points if
    needed. If the voxel_size is greater than 1, for code simplicity then the location
    is only at voxel precision.

    Args:
        array (np.array): The 3D numpy array to draw the voxels into.
        Can have singleton dimensions that will be removed and added back before being
        returned.
        x (np.array(float)): The x-coordinates of the voxel centers.
        y (np.array(float)): The y-coordinates of the voxel centers.
        z (np.array(float)): The z-coordinates of the voxel centers.
        value (float): The value to be drawn. Defaults to 1.0.
        voxel_size (int): The size of the voxel to be drawn. Rounds to the nearest
        largest odd value, as voxel centers need to be at a point.

    Returns:
        np.array. The array with voxels appplied.

    Raises:
        AssertionError: If the input array is not three-dimensional.
    """
    assert len(x) == len(y) == len(z), 'x, y and z should have the same dimensions'
    # squeeze out singleton dimensions
    original_shape = array.shape
    array = np.squeeze(array)

    assert array.ndim == 3, "Input array must be three-dimensional."

    for xi, yi, zi in zip(x, y, z):
        array = draw_voxel(array, xi, yi, zi, value=value, voxel_size=voxel_size)

    # add back singleton dimensions
    array = array.reshape(original_shape)

    return array


def draw_voxel(array, x, y, z, value: float = 1., voxel_size: int = 1):
    # get the integer part of the coordinates
    xi, yi, zi = int(x), int(y), int(z)

    # if voxel_size is 1, use trilinear interpolation
    if voxel_size == 1:
        # calculate the fractions
        fx, fy, fz = x - xi, y - yi, z - zi

        # prepare a small 3D array for the surrounding 8 voxels
        cube = np.zeros((2, 2, 2))

        # calculate weights for the corners of the cube
        weights = np.array([[[fx*fy*fz, (1-fx)*fy*fz],
                             [(1-fy)*fx*fz, (1-fx)*(1-fy)*fz]],
                            [[fx*fy*(1-fz), (1-fx)*fy*(1-fz)],
                             [(1-fy)*fx*(1-fz), (1-fx)*(1-fy)*(1-fz)]]])

        # apply the weights to the cube
        cube += value * weights

        # get the limits of the region to modify in the original array
        xmin, xmax = max(xi, 0), min(xi+2, array.shape[0])
        ymin, ymax = max(yi, 0), min(yi+2, array.shape[1])
        zmin, zmax = max(zi, 0), min(zi+2, array.shape[2])

        # apply the modification to the original array, preserving higher values
        # from nearby points
        array[xmin:xmax, ymin:ymax, zmin:zmax] = np.maximum(
            array[xmin:xmax, ymin:ymax, zmin:zmax],
            cube[xmin-xi:xmax-xi, ymin-yi:ymax-yi, zmin-zi:zmax-zi]
        )
    else:
        # if voxel_size is greater than 1, we don't need high precision,
        # so apply the voxel value directly.
        # get the limits of the region to modify in the original array
        dist_from_center = int(np.floor(voxel_size / 2))
        xmin, xmax = max(xi-dist_from_center, 0), min(xi+dist_from_center+1, array.shape[0])
        ymin, ymax = max(yi-dist_from_center, 0), min(yi+dist_from_center+1, array.shape[1])
        zmin, zmax = max(zi-dist_from_center, 0), min(zi+dist_from_center+1, array.shape[2])

        # apply the voxel value to the larger cube in the array, preserving higher values
        array[xmin:xmax, ymin:ymax, zmin:zmax] = np.maximum(
            array[xmin:xmax, ymin:ymax, zmin:zmax],
            value
        )

    return array


class Kernel():
    ''' A kernel that can return a pre-calculated kernel if it hasn't changed,
    or calculate a new one if it needs to (i.e. l or sigma have been updated by
    backprop)

    Call kernel.generate_kernel() to create a new kernel
    '''

    def __init__(self, l=5, sigma=1.):
        self.l = l
        self.sigma = sigma
        self.kernel = None
        # self.kernel = self.generate_kernel(self.l, self.sigma)
        self.kernel = self.generate_kernel_of_ones(self.l)

    def generate_kernel_of_ones(self, l=5, sigma=0.1):
        """Creates a gaussian kernel with side length `l` and a sigma of `sigma`
        if the same kernel hasn't already been calculated

        Args:
            l (int): side length of kernel
            sigma (float): standard deviation of gaussian

        Returns:
            np.array: gaussian kernel
        """
        sigma = float(sigma)

        if self.kernel is not None and self.l == l and self.sigma == sigma:
            return self.kernel


        # need to generate a new kernel

        x = np.arange(-l, l+1, 1)   # coordinate arrays that must contain 0 (the center)
        y = x.copy()
        z = x.copy()
        xx, yy, zz = np.meshgrid(x, y, z)
        self.kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*(sigma**2)))
        self.sigma = sigma
        self.l = l

        print(f'generated new kernel with sigma {self.sigma} and l {self.l}')

        return self.kernel

    def generate_kernel(self, l=5, sigma=1.):
        """Creates a gaussian kernel with side length `l` and a sigma of `sigma`
        if the same kernel hasn't already been calculated

        Args:
            l (int): side length of kernel
            sigma (float): standard deviation of gaussian

        Returns:
            np.array: gaussian kernel
        """
        sigma = float(sigma)

        if self.kernel is not None and self.l == l and self.sigma == sigma:
            return self.kernel


        # need to generate a new kernel

        x = np.arange(-l, l+1, 1)   # coordinate arrays that must contain 0 (the center)
        y = x.copy()
        z = x.copy()
        xx, yy, zz = np.meshgrid(x, y, z)
        self.kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*(sigma**2)))
        self.sigma = sigma
        self.l = l

        print(f'generated new kernel with sigma {self.sigma} and l {self.l}')

        return self.kernel


def calculate_av_label_distance(label_locations):
    """Calculates the average distance between labels in units of voxels.

    If you want the units in mm, multiply the result by .spacing of torchio.Image

    Args:
        label_locations (np.array): 3-dimensional array of label locations

    Returns:
        float: average distance between labels
    """
    label_distances = nn(label_locations)
    return np.mean(pd.DataFrame(label_distances[0]).iloc[:, 1])


def l2_loss(x):
    return torch.sum(x ** 2) / 2
