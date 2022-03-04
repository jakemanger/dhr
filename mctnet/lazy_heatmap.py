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
    


# class LazyHeatmap(tio.Image):
#     """A custom torchio.Image class that loads coordinates from a csv file (path)
#     and generates/returns a heatmap of the coordinates when loaded. It also only does so
#     when requested or needed (is lazy).
#     """

#     def __init__(self, affine, start_shape, binary=False, sigma=2, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.affine = affine
#         self.sigma = sigma
#         self.start_shape = start_shape
#         self.binary = binary


#     def generate_heatmap_from_csv(self, csv_path):
#         """Generate the heatmap from a csv file."""
#         locations = np.loadtxt(csv_path, delimiter=',', dtype=np.float).astype(int)
#         print(locations)

#         image = np.zeros(self.start_shape)
        
#         print('adding positions of corneas and rhabdoms with a gaussian kernel')

#         l = 7
#         warnings.warn('l needs to be updated!!!!! TODO')

#         print(f'Found {len(locations[:, 0])} locations')
    
#         image = apply_gaussian_kernel(
#             image,
#             (
#                 locations[:, 2],
#                 locations[:, 1],
#                 locations[:, 0]
#             ),
#             l,
#             self.sigma
#         )

#         if self.binary:
#             image[image > 0] = 1

#         return torch.from_numpy(image)


#     # def load(self) -> None:
#     #     """Generate the image using a csv file at path.

#     #         The generated data should be in the format:

#     #         Tuple containing a 4D tensor of size :math:`(C, W, H, D)` and a 2D
#     #         :math:`4 \times 4` affine matrix to convert voxel indices to world
#     #         coordinates.
#     #     """
#     #     if self._loaded:
#     #         return
#     #     paths = self.path if self._is_multipath() else [self.path]

#     #     tensor = self.generate_heatmap_from_csv(paths[0])

#     #     tensors = [tensor]
#     #     for path in paths[1:]:
#     #         new_tensor = self.generate_heatmap_from_csv(path)
#     #         if not tensor.shape[1:] == new_tensor.shape[1:]:
#     #             message = (
#     #                 f'Files shape do not match, found {tensor.shape}'
#     #                 f'and {new_tensor.shape}'
#     #             )
#     #             RuntimeError(message)
#     #         tensors.append(new_tensor)
#     #     tensor = torch.cat(tensors)
#     #     self.set_data(tensor)
#     #     self._loaded = True
