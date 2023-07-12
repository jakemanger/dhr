import torch
import torch.nn.functional as F
# import numpy as np


class GaussianKernel:
    def __init__(self, sigma=3, kernel_size=35, device='cpu', normalise=False):
        self.kernel = None
        self.device = device
        self.generate_kernel(sigma, kernel_size, normalise)

    def generate_kernel(self, sigma, kernel_size, normalise=False):
        ''' Generate a 3D Gaussian kernel.

        Only generates the kernel if it hasn't been generated or if sigma or
        kernel_size has changed.
        '''
        if (
            self.kernel is not None
            and self.sigma == sigma
            and self.kernel_size == kernel_size
        ):
            return

        with torch.no_grad():
            x, y, z = torch.meshgrid(
                [
                    torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).to(self.device),
                    torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).to(self.device),
                    torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).to(self.device)
                ],
                indexing='ij'
            )

            kernel = torch.exp(-(x**2. + y**2. + z**2.) / (2. * sigma**2.))

            # normalise
            if normalise:
                kernel.div_(torch.sum(kernel))
            # kernel = (kernel / (sigma * np.sqrt(2. * np.pi)))

        self.kernel = kernel.expand(1, 1, *kernel.size())
        self.sigma = sigma
        self.kernel_size = kernel_size

    def apply_to(self, tensor):
        ''' Apply the Gaussian kernel to a tensor. '''
        if tensor.device != self.kernel.device:
            self.kernel = self.kernel.to(tensor.device)

        tensor = tensor.squeeze(1)

        # apply the whole gaussian kernel to the tensor using 3D convolution
        tensor = F.conv3d(tensor, self.kernel, padding=self.kernel.size(2) // 2)

        tensor = tensor.unsqueeze(1)

        return tensor
