import torchio as tio
import torch
from typing import List
from torchio.transforms.augmentation.spatial import Affine
from deep_radiologist.lightning_modules import DataModule, Model
from tqdm import tqdm
import napari
# NEEDS UPDATING DUE TO CHANGES FROM torchio 0.18.71-0.18.76 from deep_radiologist.custom_gridaggregator import CustomGridAggregator


device = torch.device('cuda')

class InferenceManager:
    def __init__(
        self,
        volume_path: str,
        model: Model,
        data: DataModule,
        patch_size: int,
        patch_overlap: int,
        batch_size: int,
    ):
        """ Initialize the InferenceManager.

        Args:
            volume_path: Path to the volume to be processed.
            model: Model to be used for inference.
            data: DataModule to be used for inference.
            patch_size: Size of the patches to be processed.
            patch_overlap: Overlap of the patches.
            batch_size: Batch size for inference.
        """

        self.volume_path = volume_path
        self.model = model
        self.data = data
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.batch_size = batch_size

    def _predict(
        self,
        x_rotation: int = 0,
        y_rotation: int = 0,
        z_rotation: int = 0,
        debug_patch_plots: bool = False
    ) -> torch.Tensor:
        """ predict over a single volume at self.volume_path"""
        preprocess = self.data.get_preprocessing_transform()
        # append an affine tranformation to the preprocess variable
        affine_transform = Affine(scales=1, degrees=(x_rotation, y_rotation, z_rotation), translation=0)
        preprocess = tio.Compose([preprocess, affine_transform])

        subjects = [
            tio.Subject(
                image=tio.ScalarImage(self.volume_path, check_nans=True),
            )
        ]
        # apply transform to whole image
        print('Creating sampler and applying transform to image...')
        subjects = tio.SubjectsDataset(subjects, transform=preprocess)

        transformed = subjects[0]

        inverse_transform = transformed.get_inverse_transform(ignore_intensity=True)

        grid_sampler = tio.inference.GridSampler(
            transformed,
            self.patch_size,
            self.patch_overlap
        )

        print('Initialising patch_loader and aggregator...')
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')
        # aggregator = CustomGridAggregator(grid_sampler, overlap_mode='weighted_average') # my custom aggregator

        print('Starting inference...')
        self.model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                x = patches_batch['image'][tio.DATA].to(device)

                locations = patches_batch[tio.LOCATION]

                # if there is some feature in the first object of the batch, show it
                # for debugging purposes
                if debug_patch_plots:
                    new_x = x.clone()
                    if new_x.cpu()[0].sum() > 1:
                        for rot in range(-180, 180, 45):
                            aff_transform = Affine(scales=1, degrees=(0, 0, rot), translation=0)
                            new_x = x.clone()
                            new_x = aff_transform(tio.ScalarImage(tensor=new_x.cpu()[0]))
                            y_hat = self.model(new_x.tensor.unsqueeze(0).to(device))
                            if rot == -180:
                                viewr = napari.view_image(new_x.numpy(), name='rotation {}'.format(rot))
                            else:
                                viewr.add_image(new_x.numpy(), name='rotation {}'.format(rot))
                            viewr.add_image(y_hat.cpu().numpy(), name='prediction rotation {}'.format(rot))
                        breakpoint()
                        viewr.close()

                y_hat = self.model(x)

                aggregator.add_batch(y_hat, locations)

            out = tio.Image(tensor=aggregator.get_output_tensor(), type=tio.LABEL)
            # rotate volume back to original orientation
            print('Rotating volume back to original orientation...')
            out = inverse_transform(out)

            return out.tensor

    @staticmethod
    def _parse_combination_mode(combination_mode):
        if combination_mode not in ('max', 'average'):
            message = (
                'Overlap mode must be "max" or "average" but '
                f' "{combination_mode}" was passed'
            )
            raise ValueError(message)

    def run(
        self,
        x_rotations: List[int] = [0],
        y_rotations: List[int] = [0],
        z_rotations: List[int] = [0],
        debug_patch_plots: bool = False,
        debug_volume_plots: bool = False,
        combination_mode: str = 'average'
    ) -> tio.Image:
        """ Run multiple inferences over a volume in different directions along x, y and z axes and combine the results
        
        Args:
            x_rotations (List[int], optional): list of angles along x axis to perform inference along. Defaults to [0].
            y_rotations (List[int], optional): list of angles along y axis to perform inference along. Defaults to [0].
            z_rotations (List[int], optional): list of angles along z axis to perform inference along. Defaults to [0].
            debug_patch_plots (bool, optional): show patch plots. Defaults to False.
            debug_volume_plots (bool, optional): show volume plots. Defaults to False.
            combination_mode (str, optional): how to combine the predictions. Must be 'average' or 'max'. Defaults to 'average'.

        Returns:
            tio.Image: prediction volume
        """

        self._parse_combination_mode(combination_mode)
        self.output = None

        for r_x in tqdm(x_rotations, desc=f'Running inference in {len(x_rotations)*len(y_rotations)*len(z_rotations)} direction/s'):
            for r_y in tqdm(y_rotations, leave=False):
                for r_z in tqdm(z_rotations, leave=False):
                    print(
                        f'\nRunning inference with {r_x}° rotation along the x-axis ({list(x_rotations).index(r_x) + 1}/{len(x_rotations)}),'
                        f'\n{r_y}° rotation along the y-axis ({list(y_rotations).index(r_y) + 1}/{len(y_rotations)})'
                        f'\nand {r_z}° rotation along the z-axis ({list(z_rotations).index(r_z) + 1}/{len(z_rotations)})'
                    )

                    out = self._predict(r_x, r_y, r_z, debug_patch_plots)

                    if self.output is None:
                        print('Storing first prediction in memory...')
                        self.output = out
                    else:
                        print('Combining new prediction with previous...')
                        if combination_mode == 'average':
                            # then add the resulting tensors together
                            if debug_volume_plots:
                                viewr = napari.view_image(out.numpy(), name='out')
                                viewr.add_image(self.output.numpy(), name='self.output')

                            self.output += out

                            if debug_volume_plots:
                                viewr.add_image(self.output.numpy(), name='combined output')
                                breakpoint()
                                viewr.close()
                        elif combination_mode == 'max':
                            # then take the maximum of the tensors
                            if debug_volume_plots:
                                viewr = napari.view_image(out.numpy(), name='out')
                                viewr.add_image(self.output.numpy(), name='self.output')
                                
                            self.output = torch.maximum(self.output, out)

                            if debug_volume_plots:
                                viewr.add_image(self.output.numpy(), name='combined output')
                                breakpoint()
                                viewr.close()
        
        if combination_mode == 'average':
            # divide the resulting tensor by the number of rotations to get the average
            self.output = self.output / (len(x_rotations)*len(y_rotations)*len(z_rotations))
        
        return tio.Image(tensor=self.output, type=tio.LABEL)
