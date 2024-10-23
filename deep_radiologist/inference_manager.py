import torchio as tio
import torch
from typing import List
from torchio.transforms.augmentation.spatial import Affine
from deep_radiologist.lightning_modules import DataModule, Model
from tqdm import tqdm
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Label, PushButton
import napari
from napari.layers import Points
from deep_radiologist.image_morph import resample_by_ratio
from deep_radiologist.histogram import calculate_training_histogram, compare_histograms_and_means_sds
from deep_radiologist.heatmap_peaker import locate_peaks_in_volume


device = torch.device("cuda")


class InferenceManager:
    def __init__(
        self,
        volume_path: str,
        model: Model,
        data: DataModule,
        patch_size: int,
        patch_overlap: int,
        batch_size: int,
        resample_ratio: float = 1,
        run_debug_intensity_histogram: bool = False
    ):
        """Initialize the InferenceManager.

        Args:
            volume_path: Path to the volume to be processed.
            model: Model to be used for inference.
            data: DataModule to be used for inference.
            patch_size: Size of the patches to be processed.
            patch_overlap: Overlap of the patches.
            batch_size: Batch size for inference.
            resample_ratio: The ratio to resample by. If 1, then doesn't do anything.
        """

        self.volume_path = volume_path
        self.model = model
        self.data = data
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.batch_size = batch_size
        self.resample_ratio = resample_ratio
        self.ran_first_debug_intensity_histogram = False
        self.run_debug_intensity_histogram = run_debug_intensity_histogram

        # load image
        print('Loading volume...')
        self.img = tio.ScalarImage(self.volume_path, check_nans=True)
        # resample by ratio if provided
        if self.resample_ratio != 1:
            print(f'Resampling volume by a ratio of {self.resample_ratio}')
            self.img = resample_by_ratio(self.img, self.resample_ratio, image_interpolation='linear')


    def _predict(
        self,
        x_rotation: int = 0,
        y_rotation: int = 0,
        z_rotation: int = 0,
        debug_patch_plots: bool = False,
    ) -> torch.Tensor:
        """predict over a single volume at self.volume_path"""
        preprocess = self.data.get_preprocessing_transform()
        # append an affine tranformation to the preprocess variable
        affine_transform = Affine(
            scales=1, degrees=(x_rotation, y_rotation, z_rotation), translation=0
        )

        subjects = [
            tio.Subject(
                image=self.img,
            )
        ]

        preprocess = tio.Compose([preprocess, affine_transform])

        # apply transform to whole image
        print("Creating sampler and applying transform to image...")
        subjects = tio.SubjectsDataset(subjects, transform=preprocess)

        transformed = subjects[0]

        inverse_transform = transformed.get_inverse_transform(ignore_intensity=True)

        if not self.ran_first_debug_intensity_histogram and self.run_debug_intensity_histogram:
            # plot the intensity histogram of the transformed image against the
            # intensity histogram of the data the model was trained with
            print(
                'Plotting a debugging intensity histogram to make sure you have '
                'provided intensity values similar to the data the model was trained '
                'with. intensity values should be within the trained range for optimal '
                'performance. Close the plot to continue.'
            )
            train_histogram, train_means, train_sds, filenames = calculate_training_histogram(self.data.train_dataloader())
            compare_histograms_and_means_sds(train_histogram, train_means, train_sds, transformed.image, filenames)
            self.ran_first_debug_intensity_histogram = True


        grid_sampler = tio.inference.GridSampler(
            transformed, self.patch_size, self.patch_overlap, padding_mode='empty'
        )

        print("Initialising patch_loader and aggregator...")
        patch_loader = torch.utils.data.DataLoader(
            grid_sampler, batch_size=self.batch_size
        )
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
        # aggregator = CustomGridAggregator(grid_sampler, overlap_mode='weighted_average') # my custom aggregator

        print("Starting inference...")
        self.model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                x = patches_batch["image"][tio.DATA].to(device)

                locations = patches_batch[tio.LOCATION]

                # if there is some feature in the first object of the batch, show it
                # for debugging purposes
                if debug_patch_plots:
                    new_x = x.clone()
                    if new_x.cpu()[0].sum() > 1:
                        for rot in range(-180, 180, 45):
                            aff_transform = Affine(
                                scales=1, degrees=(0, 0, rot), translation=0
                            )
                            new_x = x.clone()
                            new_x = aff_transform(
                                tio.ScalarImage(tensor=new_x.cpu()[0])
                            )
                            y_hat = self.model(new_x.tensor.unsqueeze(0).to(device))
                            if rot == -180:
                                viewr = napari.view_image(
                                    new_x.numpy(), name="rotation {}".format(rot)
                                )
                            else:
                                viewr.add_image(
                                    new_x.numpy(), name="rotation {}".format(rot)
                                )
                            viewr.add_image(
                                y_hat.cpu().numpy(),
                                name="prediction rotation {}".format(rot),
                            )
                            input("Press enter to continue")
                        viewr.close()

                y_hat = self.model(x)

                aggregator.add_batch(y_hat, locations)

            out = tio.Image(tensor=aggregator.get_output_tensor(), type=tio.LABEL)
            # rotate volume back to original orientation
            print("Rotating volume back to original orientation...")
            out = inverse_transform(out)

            return out.tensor

    @staticmethod
    def _parse_combination_mode(combination_mode):
        if combination_mode not in ("max", "average"):
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
        debug_patch_plots: bool = True,
        debug_volume_plots: bool = True,
        average_while_running: bool = True,
        threshold: float = None
    ) -> tio.Image:
        """Run multiple inferences over a volume in different directions along x, y and z axes and return each predicted heatmap

        Args:
            x_rotations (List[int], optional): list of angles along x axis to perform inference along. Defaults to [0].
            y_rotations (List[int], optional): list of angles along y axis to perform inference along. Defaults to [0].
            z_rotations (List[int], optional): list of angles along z axis to perform inference along. Defaults to [0].
            debug_patch_plots (bool, optional): show patch plots. Defaults to False.
            debug_volume_plots (bool, optional): show volume plots. Defaults to False.
            average_while_running (bool, optional): average the images after each inference in a single direction (saves GPU memory). Defaults to True.

        Returns:
            dict[str, tio.Image]: a dictionary of x, y, z rotations and prediction volumes in that orientation.
        """

        self.outputs = {}
        self.output = None
        
        count_tensor = None

        for r_x in tqdm(
            x_rotations,
            desc=(
                "Running inference in"
                f" {len(x_rotations)*len(y_rotations)*len(z_rotations)} direction/s"
            ),
        ):
            for r_y in tqdm(y_rotations, leave=False):
                for r_z in tqdm(z_rotations, leave=False):
                    print(
                        f"\nRunning inference with {r_x}° rotation along the x-axis"
                        f" ({list(x_rotations).index(r_x) + 1}/{len(x_rotations)}),"
                        f"\n{r_y}° rotation along the y-axis"
                        f" ({list(y_rotations).index(r_y) + 1}/{len(y_rotations)})\nand"
                        f" {r_z}° rotation along the z-axis"
                        f" ({list(z_rotations).index(r_z) + 1}/{len(z_rotations)})"
                    )

                    out = self._predict(r_x, r_y, r_z, debug_patch_plots)
                    
                    if not average_while_running:
                        self.outputs[f'x: {r_x}, y: {r_y}, z: {r_z}'] = tio.Image(tensor=out, type=tio.LABEL)
                        continue

                    if threshold is not None:
                        # create a mask for values greater than the threshold
                        mask = out > threshold
                        masked_tensor = torch.where(mask, out, torch.zeros_like(out))
                        count_increment = mask.float()
                    else:
                        # no threshold, consider all values
                        masked_tensor = out
                        count_increment = torch.ones_like(out)
        
                    if self.output is None:
                        self.output = masked_tensor
                        count_tensor = count_increment
                    else:
                        self.output += masked_tensor
                        count_tensor += count_increment
                        
        if average_while_running:
            # avoid division by zero by setting any zero counts to 1
            count_tensor[count_tensor == 0] = 1
    
            average_tensor = self.output / count_tensor
    
            self.output = tio.Image(tensor=average_tensor, type=tio.LABEL)
            return self.output

        return self.outputs

    def average(self, threshold=None):
        if self.output is not None:
            print("Average was calculated while running. Returning this average.")
            return self.output
        
        if not self.outputs:
            raise ValueError('No outputs to average. Run the inference first.')

        combined_tensor = None
        count_tensor = None

        for key, output in self.outputs.items():
            tensor = output.tensor

            if threshold is not None:
                # create a mask for values greater than the threshold
                mask = tensor > threshold
                masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
                count_increment = mask.float()
            else:
                # no threshold, consider all values
                masked_tensor = tensor
                count_increment = torch.ones_like(tensor)

            if combined_tensor is None:
                combined_tensor = masked_tensor
                count_tensor = count_increment
            else:
                combined_tensor += masked_tensor
                count_tensor += count_increment

        # avoid division by zero by setting any zero counts to 1
        count_tensor[count_tensor == 0] = 1

        average_tensor = combined_tensor / count_tensor

        return tio.Image(tensor=average_tensor, type=tio.LABEL)

    def interactive_inference(self):
        """Find peaks of heatmaps interactively by choosing which predicted heatmaps to combine and what thresholds to use. 
        Plot this with napari and add a combine heatmap button.
        """

        # get the predicted heatmaps
        if self.outputs is None:
            raise ValueError('Run the inference first before calling this method')

        # create a napari viewer
        viewer = napari.Viewer()

        # add the predicted heatmaps to the viewer
        for key, value in self.outputs.items():
            viewer.add_image(value.numpy(), name=key)

        @magicgui(
            selected_layers={"widget_type": "Select", "choices": list(self.outputs.keys()), "allow_multiple": True},
            threshold={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.1},
            layout='vertical'
        )
        def combine_heatmaps(selected_layers: List[str], threshold: float):
            selected_data = [self.outputs[layer].numpy() for layer in selected_layers]

            # apply threshold to selected data
            thresholded_data = [np.where(data > threshold, data, 0) for data in selected_data]

            # combine the selected layers by averaging
            combined_heatmap = np.mean(thresholded_data, axis=0)

            # add the combined heatmap to the viewer
            layer_name = f'combined heatmap {len(viewer.layers)}'
            viewer.add_image(combined_heatmap, name=layer_name)

            # Update the choices in locate_peaks
            choices = list(self.outputs.keys()) + [layer.name for layer in viewer.layers if 'combined heatmap' in layer.name]
            locate_peaks.selected_layer.choices = choices

        @magicgui(selected_layer={"widget_type": "ComboBox", "choices": list(self.outputs.keys())},
                  peak_min_val={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.1},
                  layout='vertical')
        def locate_peaks(selected_layer: str, peak_min_val: float):
            layer = viewer.layers[selected_layer]
            heatmap = layer.data
            peaks = locate_peaks_in_volume(heatmap, min_val=peak_min_val, relative=True)
            viewer.add_points(peaks, name=f'peaks {selected_layer}')
        
        @magicgui(layout='vertical', call_button='Clear points')
        def clear_points():
            for layer in viewer.layers:
                if isinstance(layer, Points):
                    viewer.layers.remove(layer)

        @magicgui(layout='vertical', call_button='Save points')
        def save_points():
            for layer in viewer.layers:
                if isinstance(layer, Points):
                    np.save(f"{layer.name}.npy", layer.data)
            print("Points saved.")

        # Create a Label widget for the titles
        combine_title = Label(value="Combine Heatmaps")
        locate_title = Label(value="Locate Peaks in a Selected Heatmap")
        points_title = Label(value="Clear/Save Points")

        # Add the GUI widgets to the viewer with titles
        viewer.window.add_dock_widget(combine_title, area="right")
        viewer.window.add_dock_widget(combine_heatmaps, area="right")
        viewer.window.add_dock_widget(locate_title, area="right")
        viewer.window.add_dock_widget(locate_peaks, area="right")
        viewer.window.add_dock_widget(points_title, area="right")
        viewer.window.add_dock_widget(clear_points, area="right")
        viewer.window.add_dock_widget(save_points, area="right")

        # show the viewer
        napari.run()
