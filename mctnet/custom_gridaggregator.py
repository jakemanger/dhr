import torchio as tio
import torch
import warnings
import numpy as np


class CustomGridAggregator(tio.inference.GridAggregator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.average_weight = None

    @staticmethod
    def parse_overlap_mode(overlap_mode):
        if overlap_mode == 'max':
            warnings.warn('max overlap mode does not work too well, you should probably use weighted_average instead')
        if overlap_mode not in ('crop', 'average', 'weighted_average', 'max'):
            message = (
                'Overlap mode must be "crop", "average", "weighted_average" or "max" but '
                f' "{overlap_mode}" was passed'
            )
            raise ValueError(message)
    
    def add_batch(
            self,
            batch_tensor: torch.Tensor,
            locations: torch.Tensor,
            ) -> None:
        """Add batch processed by a CNN to the output prediction volume.

        Args:
            batch_tensor: 5D tensor, typically the output of a convolutional
                neural network, e.g. ``batch['image'][torchio.DATA]``.
            locations: 2D tensor with shape :math:`(B, 6)` representing the
                patch indices in the original image. They are typically
                extracted using ``batch[torchio.LOCATION]``.
        """
        batch = batch_tensor.cpu()
        locations = locations.cpu().numpy()
        self.initialize_output_tensor(batch)
        if self.overlap_mode == 'crop':
            cropped_patches, crop_locations = self.crop_batch(
                batch,
                locations,
                self.patch_overlap,
            )
            for patch, crop_location in zip(cropped_patches, crop_locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = crop_location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] = patch
        elif self.overlap_mode == 'max':
            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location

                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin
                ] = torch.max(
                    patch,
                    self._output_tensor[
                        :,
                        i_ini:i_fin,
                        j_ini:j_fin,
                        k_ini:k_fin
                    ]
                )
        elif self.overlap_mode == 'average':
            self.initialize_avgmask_tensor(batch)
            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += patch
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += 1
        elif self.overlap_mode == 'weighted_average':
            self.initialize_avgmask_tensor(batch)

            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location

                if self.average_weight is None:
                    # weight edges of patch as 0 and center as 1
                    patch_shape = patch.shape[1:]
                    assert len(np.unique(patch_shape)) == 1, 'patch size must be the same for all dimensions'
                    self.average_weight = self._make_weight_avg_kernel(
                        patch_shape[0],
                        len(patch_shape)
                    )
                    # add a batch dimension and turn into a tensor
                    self.average_weight = torch.tensor(self.average_weight[None, :, :, :])

                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += patch * self.average_weight
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += 1

                # import napari
                # napari.view_image(self._output_tensor[0, :, :, :])
                # breakpoint()

    def _make_weight_avg_kernel(self, size, n_dims):
        """Create a kernel to average patches with different weights

        That is, weight the center of the patch with 1 and the edges with 0.

        Args:
            size: Size of the kernel (one value, which is the same for width, height and, optionally, depth).
            n_dims: Number of dimensions of the kernel.
        
        Returns:
            A weighted kernel
        """

        m = size
        n = (m-1) / 2
        # make a distance matrix
        if n_dims == 2:
            x, y = np.meshgrid(np.arange(-n, n+1), np.arange(-n, n+1))
            a = np.sqrt(x**2 + y**2)
        if n_dims == 3:
            x, y, z = np.meshgrid(np.arange(-n, n+1), np.arange(-n, n+1), np.arange(-n, n+1))
            a = np.sqrt(x**2 + y**2 + z**2)

        # change to go from 0 to 1
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
        # change to go from 1 to 0
        a = 1 - a

        return a

    
