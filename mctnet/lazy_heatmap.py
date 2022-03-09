import torchio as tio
import torch
import numpy as np
from .label_generation import apply_gaussian_kernel
import warnings

class LazyHeatmapReader():

    def __init__(self, affine, start_shape, binary=False, sigma=2, l=7):
        self.affine = affine
        self.sigma = sigma
        self.start_shape = start_shape
        self.binary = binary
        self.l = l

    def generate_heatmap_from_csv(self, csv_path):
        """Generate the heatmap from a csv file."""
        locations = np.loadtxt(csv_path, delimiter=',', dtype=np.float).astype(int)

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
                self.l,
                self.sigma
            )

        if self.binary:
            image[image > 0] = 1

        return torch.from_numpy(image)

    def read(self, path):
        tensor = self.generate_heatmap_from_csv(path)
        return tensor, self.affine
    