import pytorch_lightning as pl
import torchio as tio
import os
from torch.utils.data import random_split, DataLoader
import multiprocessing
import torch
import numpy as np
from math import floor
import napari
import warnings
from scipy import spatial
from deep_radiologist.custom_unet import UNet3D
from deep_radiologist.lazy_heatmap import LazyHeatmapReader
from deep_radiologist.heatmap_peaker import locate_peaks_in_volume
from deep_radiologist.gaussian_kernel import GaussianKernel
from deep_radiologist.utils import Kernel, l2_loss
from deep_radiologist.voxel_unit_elastic_deformation import (
    VoxelUnitRandomElasticDeformation
)
from deep_radiologist.visualise_model_params import visualize_weight_distribution
from pprint import pprint


# hide warnings from pytorch complaining about num_workers=0. We are using
# a torchio.Queue with the data loader that does the multiprocessing.
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class DataModule(pl.LightningDataModule):
    """A pytorch lightning class to handle data loading and preprocessing

    Uses the torchio library to load and preprocess data and
    returns a dataloader for training and validation.

    Args:
        config (dict): dictionary containing the configuration parameters
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.train_images_dir = config["train_images_dir"]
        self.train_labels_dir = config["train_labels_dir"]
        self.test_images_dir = config["test_images_dir"]
        self.test_labels_dir = config["test_labels_dir"]
        self.histogram_landmarks_path = config["histogram_landmarks_path"]
        self.image_suffix = config["image_suffix"]
        self.label_suffix = config["label_suffix"]
        self.train_val_ratio = config["train_val_ratio"]
        self.batch_size = config["batch_size"]
        self.patch_size = config["patch_size"]
        self.samples_per_volume = config["samples_per_volume"]
        self.max_length = config["max_length"]
        self.balanced_sampler = config["balanced_sampler"]
        self.balanced_sampler_max_length = config["balanced_sampler_max_length"]
        self.ignore_empty_volumes = config["ignore_empty_volumes"]
        self.sigma = config["starting_sigma"]
        self.heatmap_max_length = config["heatmap_max_length"]
        self.learn_sigma = config["learn_sigma"]

        if str(config["num_workers"]).lower() in "auto":
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = config["num_workers"]

        if self.config["histogram_standardisation"] and not os.path.exists(
            self.histogram_landmarks_path
        ):
            self.create_histogram_landmarks()

        if self.balanced_sampler and config['relative_heatmap_peak']:
            warnings.warn(
                'Localised point metrics will be incorrect as Heatmap relative peak'
                'inference only works with balanced_sampler being False, as coordinates'
                'will be localised on empty frames. Note, this will not affect training '
                'as coordinate localisation metrics are not used in the loss function.'
            )

        if not self.balanced_sampler and self.balanced_sampler_max_length < (config['patch_size'] / 2):
            raise ValueError(
                'balanced_sampler_max_length must be greater than or equal to half of patch_size'
                'if balanced_sampler is False. Otherwise, the sampler will not be able to'
                'sample patches if coordinates are at the edge of the volume.'
            )

    def get_max_shape(self, subjects):
        """Gets the maximum shape of the images in a list of subjects

        Args:
            subjects (list): list of subjects

        Returns:
            max_shape (tuple): maximum shape of the images in the list of
            subjects
        """

        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def _find_data_filenames(self, image_dir, label_dir):
        """Finds the filenames of the images and labels in the given directories

        If self.ignore_empty_volumes is True, then only returns files with
        at least one label.

        Args:
            image_dir (str): path to directory containing images
            label_dir (str): path to directory containing labels

        Returns:
            image_filenames (list): list of common prefix of image and label filenames
        """

        images = []
        labels = []
        for file in os.listdir(image_dir):
            if file.endswith(".nii"):
                spltstr = file.split("-")
                if len(spltstr) == 3:
                    images.append(spltstr[0] + "-" + spltstr[1])
                else:
                    images.append(spltstr[0])
        for file in os.listdir(label_dir):
            if file.endswith(".csv") and self.label_suffix in file:
                if (
                    not self.ignore_empty_volumes
                    or os.path.getsize(label_dir + file) > 0
                ):
                    spltstr = file.split("-")
                    if len(spltstr) == 3:
                        labels.append(spltstr[0] + "-" + spltstr[1])
                    else:
                        labels.append(spltstr[0])

        filenames = sorted(list(set(images) & set(labels)))
        print(
            f"Found {len(filenames)} labelled images with labels in {image_dir} and"
            f" {label_dir} for analysis"
        )
        return filenames

    def _load_subjects(self, image_dir, label_dir):
        """Loads the images and labels from the given directories

        Args:
            image_dir (str): path to directory containing images
            label_dir (str): path to directory containing labels

        Returns:
            subjects (list): list of subjects (with images and labels)
        """

        subjects = []
        # find all the .nii files
        filenames = self._find_data_filenames(image_dir, label_dir)

        self.gk = None
        if not self.config['learn_sigma']:
            self.gk = GaussianKernel(
                self.config['starting_sigma'],
                self.config['heatmap_max_length'],
                device=torch.device('cpu')
            )

        # now add them to a list of subjects
        for filename in filenames:
            nm_comps = filename.split("-")

            if len(nm_comps) == 2:
                # name includes cropped patch ID
                path = f'{nm_comps[0]}-{nm_comps[1]}'
            else:
                path = f'{nm_comps[0]}'

            img = tio.ScalarImage(
                f"{image_dir}{path}-{self.image_suffix}.nii",
                check_nans=True,
            )

            heatmap_reader = LazyHeatmapReader(
                affine=img.affine,
                start_shape=img.shape,
                value=self.config['heatmap_scalar'] if 'heatmap_scalar' in self.config else 1.,
                gaussian_kernel=self.gk,
                subpix_accuracy=self.config['subpix_accuracy'] if 'subpix_accuracy' in self.config else False
            )
            lbl = tio.Image(
                path=f"{label_dir}{path}-{self.label_suffix}.csv",
                type=tio.LABEL,
                check_nans=True,
                reader=heatmap_reader.read,
            )
            smpl_map_reader = LazyHeatmapReader(
                affine=img.affine,
                start_shape=img.shape,
                voxel_size=self.balanced_sampler_max_length*2
            )
            smpl_map = tio.Image(
                path=f"{label_dir}{path}-{self.label_suffix}.csv",
                type=tio.LABEL,
                check_nans=True,
                reader=smpl_map_reader.read,
            )

            subject = tio.Subject(
                image=img, label=lbl, sampling_map=smpl_map, filename=filename
            )
            if self.config['debug_sampling_plots'] if 'debug_sampling_plots' in self.config else False:
                viewer = napari.view_image(
                    img.data.numpy(),
                    name='image',
                )
                viewer.add_image(
                    lbl.data.numpy(),
                    name='label before any heatmap thresholding (if applied). See debug plots during training to see effect of heatmap thresholding.',
                )
                viewer.add_image(
                    smpl_map.data.numpy(),
                    name='sampling_map',
                )
                viewer.add_image(
                    np.ones(
                        (self.config['patch_size'],)*3,
                    ),
                    name='example patch',
                    colormap='green',
                )

                input(f'Viewing sampling plots of {filename}. Press enter to continue.')
                viewer.close()

            subjects.append(subject)
        return subjects

    def get_preprocessing_transform(self):
        """Returns the preprocessing transform for the dataset

        Returns:
            transform (torchvision.transforms.Compose): preprocessing transform
        """
        preprocess_list = [tio.ToCanonical(), tio.EnsureShapeMultiple(8)]

        if self.config["histogram_standardisation"]:
            preprocess_list.append(
                tio.HistogramStandardization(
                    {
                        "default_image_name": self.histogram_landmarks_path,
                        f"{self.image_suffix}": self.histogram_landmarks_path,
                    },
                    masking_method=tio.ZNormalization.mean,
                )
            )

        if self.config["z_normalisation"]:
            preprocess_list.append(
                tio.ZNormalization(masking_method=tio.ZNormalization.mean)
            )

        preprocess_list.append(tio.RescaleIntensity(out_min_max=(0, 1)))

        return tio.Compose(preprocess_list)

    def get_augmentation_transform(self):
        """Returns the augmentation transform for the dataset

        Returns:
            transform (torchio.Compose): augmentation transform
        """

        if 'random_affine_rotation_range' in self.config:
            rotation_range = self.config["random_affine_rotation_range"]
        else:
            rotation_range = (
                self.config["random_affine_rotation_range_x"],
                self.config["random_affine_rotation_range_y"],
                self.config["random_affine_rotation_range_z"],
            )

        augment_list = [
            tio.RandomAffine(
                p=self.config["random_affine_prob"],
                scales=self.config["random_affine_scale_range"],
                degrees=rotation_range,
                translation=self.config["random_affine_translation_range"],
            ),
            VoxelUnitRandomElasticDeformation(
                p=self.config["random_elastic_deformation_prob"],
                num_control_points=self.config[
                    "random_elastic_deformation_num_control_points"
                ],
                max_displacement=self.config[
                    "random_elastic_deformation_max_displacement"
                ],
            ),
        ]

        if self.config['random_log_gamma'] > 0:
            augment_list.append(
                tio.RandomGamma(self.config['random_log_gamma'])
            )

        augment = tio.Compose(augment_list)

        return augment

    def get_sampler(self):
        if self.balanced_sampler is True:
            self.sampler = tio.LabelSampler(
                patch_size=self.patch_size,
                label_name="sampling_map",
                label_probabilities={0: 0.5, 1: 0.5},
            )
        elif self.balanced_sampler is False:
            self.sampler = tio.UniformSampler(patch_size=self.patch_size)
        elif isinstance(self.balanced_sampler, (int, float)):
            self.sampler = tio.LabelSampler(
                patch_size=self.patch_size,
                label_name="sampling_map",
                label_probabilities={0: 1 - self.balanced_sampler, 1: self.balanced_sampler},
            )
        else:
            raise ValueError(
                "balanced_sampler must be a boolean or a float between 0 and 1"
            )

    def create_histogram_landmarks(self):
        """Create histogram landmarks for the dataset.

        This is used to normalise the images. Is saved to the root directory.
        """

        print("Histogram landmarks not found, creating them...")

        # find all the .nii files
        filenames = self._find_data_filenames(
            self.train_images_dir, self.train_labels_dir
        )
        filenames = [
            self.train_images_dir + f + f"-{self.image_suffix}.nii"
            for f in filenames
        ]

        landmarks = tio.HistogramStandardization.train(
            filenames,
            output_path=self.histogram_landmarks_path,
            masking_function=tio.ZNormalization.mean,
        )
        print(f"Histogram landmarks saved to {self.histogram_landmarks_path}")
        np.set_printoptions(suppress=True, precision=3)
        print("\nTrained landmarks:", landmarks)

    def setup(self, stage=None):
        """Sets up the dataset.

        Args:
            stage (str): stage to setup the dataset for (used internally by pytorch lightning)
                should be either 'fit' or 'test'
        """
        print("Running setup!")

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        if stage == "fit" or stage is None:
            self.subjects = self._load_subjects(
                self.train_images_dir, self.train_labels_dir
            )
            num_subjects = len(self.subjects)
            num_train_subjects = int(floor(num_subjects * self.train_val_ratio))
            num_val_subjects = num_subjects - num_train_subjects
            splits = num_train_subjects, num_val_subjects
            train_subjects, val_subjects = random_split(self.subjects, splits)
            self.train_set = tio.SubjectsDataset(
                train_subjects, transform=self.transform
            )
            self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)

        if stage == "test" or stage is None:
            self.test_subjects = self._load_subjects(
                self.test_images_dir, self.test_labels_dir
            )
            self.test_set = tio.SubjectsDataset(
                self.test_subjects, transform=self.preprocess
            )

        self.get_sampler()

    def train_dataloader(self):
        # print('Creating train dataloader')
        # print(f'learn sigma is {self.learn_sigma}')

        self.train_queue = tio.Queue(
            subjects_dataset=self.train_set,
            max_length=self.max_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.num_workers,
        )
        # num_workers refers to the number of workers used to load and transform the volumes.
        # Multiprocessing is not needed to pop patches from the queue, so you should always use
        # num_workers=0 for the DataLoader you instantiate to generate training batches.
        return DataLoader(
            self.train_queue,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=self.config['pin_memory'] if 'pin_memory' in self.config else False,
        )

    def val_dataloader(self):
        # print('Creating val dataloader')

        self.val_queue = tio.Queue(
            subjects_dataset=self.val_set,
            max_length=self.max_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.num_workers,
        )
        return DataLoader(
            self.val_queue,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=self.config['pin_memory'] if 'pin_memory' in self.config else False,
        )

    def test_dataloader(self):
        # print('creating test dataloader')

        self.test_queue = tio.Queue(
            subjects_dataset=self.test_set,
            max_length=self.max_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.num_workers
        )
        return DataLoader(
            self.test_queue,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=self.config['pin_memory'] if 'pin_memory' in self.config else False,
        )


class Model(pl.LightningModule):
    """Model class for the deep_radiologist network.

    Setup for use with pytorch Lightning.

    Args:
        config (dict): configuration dictionary (i.e. hyperparameters)
    """

    def __init__(self, config):
        super().__init__()

        pprint(f"Initiating model using the following config: {config}")

        self._model = UNet3D(
            in_channels=1,
            out_classes=1,
            num_encoding_blocks=config["num_encoding_blocks"],
            out_channels_first_layer=config["out_channels_first_layer"],
            normalization="batch",
            pooling_type=config["pooling_type"],
            upsampling_type=config["upsampling_type"],
            padding=True,
            activation=config["act"],
            dimensions=3,
            dropout=config["dropout"],
            output_activation=config["output_activation"],
            double_channels_with_depth=config['double_channels_with_depth'] if 'double_channels_with_depth' in config else True,
            softargmax=config['softargmax'],
            learn_sigma=config['learn_sigma'],
            starting_sigma=float(config['starting_sigma']) if config['learn_sigma'] else None
        )
        # self._model = UNet3DWithSigma(
        #     in_channels=1,
        #     out_channels=1,
        #     final_sigmoid=False,
        #     f_maps=config['out_channels_first_layer'],
        #     num_groups=8,
        #     num_levels=config['num_encoding_blocks'],
        #     layer_order='gcr',
        #     is_segmentation=False,
        #     learn_sigma=config['learn_sigma'],
        #     starting_sigma=float(config['starting_sigma']) if config['learn_sigma'] else None
        # )

        self.criterion = torch.nn.MSELoss()

        self.config = config

        self.debug_plots = config["debug_plots"]

        if (
            ('heatmap_min_threshold' in config and config['heatmap_min_threshold'] != None)
            or ('heatmap_max_threshold' in config and config['heatmap_max_threshold'] != None)
        ):
            if 'heatmap_min_threshold' in config:
                self.heatmap_min_threshold = config['heatmap_min_threshold']
            else:
                self.heatmap_min_threshold = None
            if 'heatmap_max_threshold' in config:
                self.heatmap_max_threshold = config['heatmap_max_threshold']
            else:
                self.heatmap_max_threshold = None
            self.use_heatmap_thresholding = True
        else:
            self.use_heatmap_thresholding = False

        if config["visualise_model"]:
            pprint(self._model)
            # visualize_weight_distribution(self._model)
            # visualize_weight_distribution(self._model.encoder.encoding_blocks[0])
            # visualize_weight_distribution(self._model.decoder)

        if config['learn_sigma']:
            assert 'sigma_regularizer' in config, (
                'The loss function for learning optimal sigma values requires a '
                '`sigma_regularizer` hyperparameter in your config file.'
            )
            self.sigma_regularizer = float(config['sigma_regularizer'])

        self.gk = GaussianKernel(
            self.config['starting_sigma'],
            self.config['heatmap_max_length'],
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            normalise=True
        )
            # self.sigmoid = torch.nn.Sigmoid()

        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.config['optimiser'] == 'SGD':
            optimizer = torch.optim.SGD(
                self._model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay'],
                momentum=self.config['momentum'] if 'momentum' in self.config else 0,
                nesterov=self.config['nesterov'] if 'nesterov' in self.config else 0
            )
        elif self.config['optimiser'] == 'Adam':
            optimizer = torch.optim.Adam(
                self._model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise NotImplementedError(
                f"{self.config['optimiser']} optimiser has not been implemented."
            )
        return optimizer

    def _apply_gaussian(self, tensor):
        ''' applies a gaussian kernel if learning the sigma
        Note, this is experimental.

        Otherwise, it will have been applied on the cpu previously.
        '''
        if self.config['learn_sigma']:
            sigma = self._model.sigma
            self.gk.generate_kernel(
                sigma,
                self.config['heatmap_max_length'],
                normalise=True
            )
            self.log("sigma", sigma, batch_size=self.config['batch_size'], sync_dist=True)
            # apply gaussian distribution to label at points
            # by convolving a torch gaussian kernel
            tensor = self.gk.apply_to(tensor)

        return tensor

    def prepare_batch(self, batch):
        image = batch['image'][tio.DATA]
        label = batch['label'][tio.DATA]
        return image, self._apply_gaussian(label)

    def apply_heatmap_thresholding(self, x, y):
        if self.heatmap_min_threshold is not None and self.heatmap_max_threshold is not None:
            mask = (x <= self.heatmap_min_threshold) | (x >= self.heatmap_max_threshold)
        elif self.heatmap_min_threshold is not None:
            mask = (x <= self.heatmap_min_threshold)
        elif self.heatmap_max_threshold is not None:
            mask = (x >= self.heatmap_max_threshold)
        else:
            # raise an appropriate Exception
            raise ValueError(
                'apply_heatmap_thresholding() requires a minimum or maximum threshold '
                'to work correctly.'
            )

        y[mask] = 0
        return x, y

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)

        if self.use_heatmap_thresholding:
            x, y = self.apply_heatmap_thresholding(x, y)

        y_hat = self.forward(x)

        if self.debug_plots:
            self.viewer = napari.view_image(x.cpu().numpy(), name="Input")
            self.viewer.add_image(y.cpu().numpy(), name="Ground Truth")
            self.viewer.add_image(y_hat.cpu().detach().numpy(), name="Prediction")
            # # test elastic deformation effect
            # augmentation = (
            #     VoxelUnitRandomElasticDeformation(
            #         p=self.config["random_elastic_deformation_prob"],
            #         num_control_points=self.config["random_elastic_deformation_num_control_points"],
            #         max_displacement=self.config["random_elastic_deformation_max_displacement"],
            #     ),
            # )

            # transform = tio.Compose(
            #     augmentation
            # )
            # self.viewer.add_image(transform(x[0].cpu()).numpy(), name='augmented x')

        return y_hat, y

    def forward(self, x):
        y_hat = self._model(x)
        return y_hat

    def _calculate_loss(self, y_hat, y):
        # if self.config['loss_in_sigma_space']:
        #     mask = y != 0
        #     return self.criterion(y_hat * mask, y)
        # else:
        loss = self.criterion(y_hat, y)

        if not self.config['learn_sigma']:
            return loss

        sigma_loss = l2_loss(self._model.sigma) * self.sigma_regularizer

        return loss + sigma_loss

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self._calculate_loss(y_hat, y)

        if self.debug_plots or self.config['mse_with_f1']:
            tp, fp, fn, failures, mean_loc_err = self.calc_acc(y_hat, y)

        if self.config['mse_with_f1']:
            f1 = 1
            if (tp + fp + fn) != 0: # avoid divide by zero errors if there are no features in volume
                f1 = (2 * tp / (2 * tp + fp + fn))

            self.log("train_1_take_f1", np.float32(1 - f1), batch_size=self.config["batch_size"], sync_dist=True)
            loss = loss + np.float32(1 - f1)


        self.log(
            "train_loss", loss, prog_bar=True, batch_size=self.config["batch_size"], sync_dist=True
        )


        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self._calculate_loss(y_hat, y)

        tp, fp, fn, failures, mean_loc_err = self.calc_acc(y_hat, y)

        f1 = 1
        # avoid divide by zero errors if there are no features in volume
        if (tp + fp + fn) != 0:
            f1 = (2 * tp / (2 * tp + fp + fn))

        if self.config['mse_with_f1']:
            loss = loss + np.float32(1 - f1)

        self.log("val_loss", loss, batch_size=self.config["batch_size"], sync_dist=True)
        self.log("val_1_take_f1", np.float32(1 - f1), batch_size=self.config["batch_size"], sync_dist=True)
        self.log("val_tp", tp, batch_size=self.config["batch_size"], sync_dist=True)
        self.log("val_fp", fp, batch_size=self.config["batch_size"], sync_dist=True)
        self.log("val_fn", fn, batch_size=self.config["batch_size"], sync_dist=True)
        self.log("val_failures", failures, batch_size=self.config["batch_size"], sync_dist=True)
        self.log("val_mean_loc_err", mean_loc_err, batch_size=self.config["batch_size"], sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self._calculate_loss(y_hat, y)

        tp, fp, fn, failures, mean_loc_err = self.calc_acc(y_hat, y)

        f1 = 1
        # avoid divide by zero errors if there are no features in volume
        if (tp + fp + fn) != 0:
            f1 = (2 * tp / (2 * tp + fp + fn))

        if self.config['mse_with_f1']:
            loss = loss + np.float32(1 - f1)

        self.log("test_1_take_f1", np.float32(1 - f1), batch_size=self.config["batch_size"], sync_dist=True)
        self.log("test_loss", loss, batch_size=self.config["batch_size"], sync_dist=True)

        return loss

    def _locate_coords(self, heatmap, min_val=None):
        if min_val is None:
            min_val = self.config["peak_min_val"]

        coords = locate_peaks_in_volume(
            heatmap, min_val=min_val, relative=self.config['relative_heatmap_peak']
        )
        return coords

    def _get_acc_metrics(self, y_hat, y, k=3):
        """Calculates accuracy metrics for a set of predicted and ground truth coordinates.

        Is a true positive if the distance between the predicted and closest ground truth coordinate
        is less than the correct_prediction_distance config parameter and that ground truth coordinate
        doesn't already have a better matching prediction (tested up to k closest matches). Is a false
        positive if the distance is greater than the correct_prediction_distance parameter or it
        already has a closer true positive. Is a false negative if the ground truth does not have a
        corresponding true positive.

        Args:
            y_hat (np.ndarray): predicted coordinates
            y (np.ndarray): ground truth coordinates

        Returns:
            tp (float): true positives
            fp (float): false positives
            fn (float): false negatives
            loc_errs (np.ndarray): location errors
        """

        if len(y) > 0 and len(y_hat) > 0:
            tree = spatial.cKDTree(y_hat)
            closest_dists, closest_nbrs = tree.query(y, k=k)

            y_match = list()
            y_hat_match = list()
            dists = list()

            for i in range(k):
                nbrs_k = closest_nbrs[:, i]
                dists_k = closest_dists[:, i]

                # sort by closest distance
                sort_idx = np.argsort(dists_k)
                nbrs_k = nbrs_k[sort_idx]
                dists_k = dists_k[sort_idx]

                for j in range(len(nbrs_k)):
                    if j not in y_hat_match and y[j] not in y_match:
                        y_hat_match.append(j)
                        y_match.append(y[j])
                        dists.append(dists_k[j])
        else:
            dists = []

        dists = np.array(dists)

        tp = len(dists[dists < self.config["correct_prediction_distance"]])
        fp = len(y_hat) - tp
        fn = len(y) - tp

        loc_errors = dists[dists < self.config["correct_prediction_distance"]]

        if len(loc_errors) == 0:
            loc_errors = np.array([0])

        return tp, fp, fn, loc_errors

    def calc_acc(self, y_hats, ys):
        """Calculates accuracy metrics to be saved for the batch

        NOTE: these are approximate, as ground truth coordinates are computed from the y (groundtruth)
        heatmap by finding its peaks and NOT the original coordinates in the csv file.
        This is to let the coordinates be updated with augmentation to labels and to make it a fair
        comparison when locating areas along borders of the patch. This shouldn't impact accuracy
        scores too much, however, and should be suitable for hyperparameter tuning. Final accuracy
        should be calculated using the coordinates in the file's csv after training (with no
        augmentation). I do this on the entire volume (not patches used for training/tuning).

        Averages the accuracy metrics across the batch.

        Args:
            y_hats (torch.Tensor): predicted heatmap
            ys (torch.Tensor): ground truth heatmap

        Returns:
            tp (int): true positives
            fp (int): false positives
            fn (int): false negatives
            failures (int): number of failed predictions (fp + fn)
            mean_loc_err (float): mean location error in voxels
        """

        tps = []
        fps = []
        fns = []
        loc_errs = []
        y_coords = []

        for y_hat, y in zip(y_hats, ys):
            y_hat_coord = self._locate_coords(y_hat.cpu().detach().numpy())
            y_coord = self._locate_coords(y.cpu().detach().numpy())
            tp, fp, fn, loc_err = self._get_acc_metrics(y_hat_coord, y_coord)

            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            loc_errs.append(loc_err)
            if self.debug_plots:
                y_coords.append(y_coord)

        tp = np.mean(tps)
        fp = np.mean(fps)
        fn = np.mean(fns)

        failures = fp + fn

        if len(loc_errs) > 1:
            loc_err = np.mean([np.mean(i) for i in loc_errs])
        else:
            loc_err = np.mean(loc_errs)

        # import napari
        # viewr = napari.view_image(y.cpu().detach().numpy())
        if self.debug_plots and self.viewer is not None:
            for i, y_coord in enumerate(y_coords):
                self.viewer.add_points(
                    y_coord,
                    name=f"Ground truth volume {i} in batch",
                    size=2,
                    face_color="blue",
                )
            input("Press enter to continue...")
            self.viewer.close()

        return (
            tp.astype(np.float32),
            fp.astype(np.float32),
            fn.astype(np.float32),
            failures.astype(np.float32),
            loc_err.astype(np.float32),
        )
