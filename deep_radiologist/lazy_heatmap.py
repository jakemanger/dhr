import torch
import numpy as np
from .utils import draw_voxels


class LazyHeatmapReader():
    """ Lazily reads in 3d label locations from a csv file and then
    generates a heatmap from them.

    Allows the clever loading and memory management features of
    torchio for heatmap regression.
    """

    def __init__(self, affine, start_shape, voxel_size=1, value=1., gaussian_kernel=None):
        self.affine = affine
        self.start_shape = start_shape
        self.voxel_size = voxel_size
        self.value = value
        if gaussian_kernel is not None:
            self.gk = gaussian_kernel

    def generate_heatmap_from_csv(self, csv_path):
        """Generate the heatmap from a csv file."""
        locations = np.loadtxt(
            csv_path,
            delimiter=',',
            ndmin=2,
            dtype=np.float64
        ).astype(int)

        image = np.zeros(self.start_shape)

        if (len(locations) > 0):
            image = draw_voxels(
                image,
                locations[:, 0],
                locations[:, 1],
                locations[:, 2],
                self.value,
                self.voxel_size
            )

        image = torch.from_numpy(image.astype(np.float16))

        if (len(locations) > 0):
            if self.gk:
                self.gk.apply_to(image)

        return image

    def read(self, path):
        self.tensor = self.generate_heatmap_from_csv(path)
        # print(f'Generating lazy heatmap using a kernel with {self.kernel.sigma}')
        return self.tensor, self.affine
