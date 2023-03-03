import torchio as tio
import torch
import numpy as np
from .label_generation import apply_gaussian_kernel
import warnings

class LazyHeatmapReader():
    """ Lazily reads in 3d label locations from a csv file and then generates a heatmap from them.
    
    Allows the clever loading and memory management features of torchio for heatmap regression.
    
    """

    def __init__(self, affine, start_shape, kernel, l):
        self.affine = affine
        self.start_shape = start_shape
        self.kernel = kernel
        self.l = l

    def generate_heatmap_from_csv(self, csv_path):
        """Generate the heatmap from a csv file."""
        locations = np.loadtxt(csv_path, delimiter=',', ndmin=2, dtype=np.float).astype(int)

        image = np.zeros(self.start_shape)

        # print('adding positions of corneas and rhabdoms with a gaussian kernel')

        # print(f'Found {len(locations[:, 0])} locations')

        if (len(locations) > 0):
            image = apply_gaussian_kernel(
                image,
                (
                    locations[:, 0],
                    locations[:, 1],
                    locations[:, 2]
                ),
                self.kernel,
                self.l
            )

        return torch.from_numpy(image.astype(np.float32))

    def read(self, path):
        tensor = self.generate_heatmap_from_csv(path)
        return tensor, self.affine
