import torch
import numpy as np
from .utils import draw_voxels
from .label_generation import apply_kernel
import time

class LazyHeatmapReader():
    """ Lazily reads in 3d label locations from a csv file and then
    generates a heatmap from them.

    Allows the clever loading and memory management features of
    torchio for heatmap regression.
    """

    def __init__(self, affine, start_shape, voxel_size=1, value=1., gaussian_kernel=None, subpix_accuracy=False, sigma=None):
        self.affine = affine
        self.start_shape = start_shape
        self.voxel_size = voxel_size
        self.value = value
        self.subpix_accuracy = subpix_accuracy
        self.gk = None
        self.sigma = sigma

        if gaussian_kernel is not None:
            self.gk = gaussian_kernel

    def generate_heatmap_from_csv(self, csv_path):
        """Generate the heatmap from a csv file."""
        locations = np.loadtxt(
            csv_path,
            delimiter=',',
            ndmin=2,
            dtype=np.float32
        )

        if not self.subpix_accuracy:
            start_time = time.time()
            image = np.zeros(self.start_shape)
            # use a gaussian kernel to generate the heatmap
            assert self.gk is not None, 'a kernel is required if not using subpix accuracy formula gaussian approach. Make sure gk is supplied'
            # set to pixel accuracy
            locations = locations.astype(int)

            # apply kernel to locations very quickly using indexing
            image = apply_kernel(
                image,
                (
                    locations[:, 0],
                    locations[:, 1],
                    locations[:, 2]
                ),
                self.gk.kernel.squeeze().detach().cpu().numpy(),
            )
            image = torch.from_numpy(image.astype(np.float32))
            # import napari
            # viewer = napari.view_image(image.numpy())
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
            return image

        # otherwise go with formula approach
        start_time = time.time()
        start_shape = self.start_shape
        image = torch.zeros(start_shape)

        # Generate grid coordinates for the volume
        x = torch.arange(start_shape[1]).float().unsqueeze(-1).unsqueeze(-1)
        y = torch.arange(start_shape[2]).float().unsqueeze(0).unsqueeze(-1)
        z = torch.arange(start_shape[3]).float().unsqueeze(0).unsqueeze(0)

        # Convert locations to torch tensor
        locations = torch.tensor(locations, dtype=torch.float32)

        # Calculate distance from each point to each peak using broadcasting
        distance = torch.sqrt((x - locations[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)) ** 2 +
                              (y - locations[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)) ** 2 +
                              (z - locations[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)) ** 2)

        # Apply Gaussian function to compute heatmap
        gauss = torch.exp(-distance / (2 * self.sigma ** 2))

        # Sum over peaks to get the final heatmap
        image = torch.max(gauss, dim=0).values * self.value

        # Visualize the heatmap using Napari
        # import napari
        # viewer = napari.view_image(image.numpy())
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        return image.unsqueeze(0)  # add batch dim

        # otherwise go with slower convolutional approach
        # if (len(locations) > 0):
        #     image = draw_voxels(
        #         image,
        #         locations[:, 0],
        #         locations[:, 1],
        #         locations[:, 2],
        #         self.value,
        #         self.voxel_size
        #     )
        # image = torch.from_numpy(image.astype(np.float32))
        # if (len(locations) > 0):
        #     if self.gk:
        #         image = self.gk.apply_to(image)
        # return image

    def read(self, path):
        self.tensor = self.generate_heatmap_from_csv(path)
        # print(f'Generating lazy heatmap using a kernel with {self.kernel.sigma}')
        return self.tensor, self.affine
