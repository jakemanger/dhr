import torch
import numpy as np
from .utils import draw_voxels


class LazyHeatmapReader():
    """ Lazily reads in 3d label locations from a csv file and then
    generates a heatmap from them.

    Allows the clever loading and memory management features of
    torchio for heatmap regression.
    """

    def __init__(self, affine, start_shape, voxel_size=1):
        self.affine = affine
        self.start_shape = start_shape
        self.voxel_size = voxel_size

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
                1.,
                self.voxel_size
            )

        image = torch.from_numpy(image.astype(np.float32))

        # import napari
        # viewer = napari.view_image(image.cpu().numpy(), name='image')
        # viewer.add_points(
        #     locations,
        #     name="Coordinates",
        #     size=2,
        #     face_color="blue",
        # )
        # input(f'viewing image with voxel_size of {self.voxel_size}')

        return image

    def read(self, path):
        self.tensor = self.generate_heatmap_from_csv(path)
        # print(f'Generating lazy heatmap using a kernel with {self.kernel.sigma}')
        return self.tensor, self.affine
