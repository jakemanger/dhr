from docstring_parser import compose
import pytorch_lightning as pl
import torchio as tio
import os
from torch.utils.data import random_split, DataLoader
import multiprocessing
import torch
import numpy as np
import napari
import warnings
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from deep_radiologist.custom_unet import UNet3D
from deep_radiologist.image_morph import crop_3d_coords
from deep_radiologist.lazy_heatmap import LazyHeatmapReader
from deep_radiologist.heatmap_peaker import locate_peaks_in_volume
from deep_radiologist.utils import generate_kernel
from deep_radiologist.voxel_unit_elastic_deformation import VoxelUnitRandomElasticDeformation

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

        if self.learn_sigma:
            raise NotImplementedError("Sigma learning not implemented yet")

        if self.config["histogram_standardisation"] and not os.path.exists(
            self.histogram_landmarks_path
        ):
            self.create_histogram_landmarks()

    def get_max_shape(self, subjects):
        """Gets the maximum shape of the images in a list of subjects

        Args:
            subjects (list): list of subjects

        Returns:
            max_shape (tuple): maximum shape of the images in the list of subjects
        """

        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def _find_data_filenames(self, image_dir, label_dir):
        """Finds the filenames of the images and labels in the given directories

        If self.ignore_empty_volumes is True, then only returns files with at least one label.

        Args:
            image_dir (str): path to directory containing images
            label_dir (str): path to directory containing labels

        Returns:
            image_filenames (list): list of common prefix of image and label filenames
        """

        images = []
        labels = []
        for file in os.listdir(image_dir):
            if file.endswith(".nii.gz"):
                spltstr = file.split("-")
                images.append(spltstr[0] + "-" + spltstr[1])
        for file in os.listdir(label_dir):
            if file.endswith(".csv") and self.label_suffix in file:
                if (
                    not self.ignore_empty_volumes
                    or os.path.getsize(label_dir + file) > 0
                ):
                    spltstr = file.split("-")
                    labels.append(spltstr[0] + "-" + spltstr[1])

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

        kernel = generate_kernel(l=self.heatmap_max_length, sigma=self.sigma)

        # now add them to a list of subjects
        for filename in filenames:
            nm_comps = filename.split("-")
            img = tio.ScalarImage(
                f"{image_dir}{nm_comps[0]}-{nm_comps[1]}-{self.image_suffix}.nii.gz",
                check_nans=True,
            )

            heatmap_reader = LazyHeatmapReader(
                affine=img.affine,
                start_shape=img.shape,
                kernel=kernel,
                l=self.heatmap_max_length,
            )
            lbl = tio.Image(
                path=f"{label_dir}{nm_comps[0]}-{nm_comps[1]}-{self.label_suffix}.csv",
                type=tio.LABEL,
                check_nans=True,
                reader=heatmap_reader.read,
            )
            reader = LazyHeatmapReader(
                affine=img.affine,
                start_shape=img.shape,
                kernel=kernel,
                l=self.heatmap_max_length,
                binary=True,
            )
            smpl_map = tio.Image(
                path=f"{label_dir}{nm_comps[0]}-{nm_comps[1]}-{self.label_suffix}.csv",
                type=tio.LABEL,
                check_nans=True,
                reader=reader.read,
            )

            # import napari
            # viewr = napari.view_image(img.numpy()) # testing only
            # viewr.add_image(lbl.numpy()) # testing only
            # viewr.add_image(smpl_map.numpy()) # testing only
            # breakpoint()

            subject = tio.Subject(
                image=img, label=lbl, sampling_map=smpl_map, filename=filename
            )
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

        return tio.Compose(preprocess_list)

    def get_augmentation_transform(self):
        """Returns the augmentation transform for the dataset

        Returns:
            transform (torchvision.transforms.Compose): augmentation transform
        """

        augment = tio.Compose(
            [
                tio.RandomAffine(
                    p=self.config["random_affine_prob"],
                    scales=self.config["random_affine_scale_range"],
                    degrees=self.config["random_affine_rotation_range"],
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
        )
        return augment

    def get_sampler(self):
        if self.balanced_sampler:
            self.sampler = tio.LabelSampler(
                patch_size=self.patch_size,
                label_name="sampling_map",
                label_probabilities={0: 0.5, 1: 0.5},
            )
        else:
            self.sampler = tio.UniformSampler(patch_size=self.patch_size)

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
            self.train_images_dir + f + f"-{self.image_suffix}.nii.gz"
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
            num_train_subjects = int(round(num_subjects * self.train_val_ratio))
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

    def _update_sigma(self):
        raise NotImplementedError("Sigma learning not implemented yet")
        # note, this will need to reload the images with a new kernel
        # self.sigma = self.trainer.model.sigma
        # self.sigma = 2
        # print(f'Sigma has been updated to {self.sigma}')

    def train_dataloader(self):
        # print('Creating train dataloader')
        # print(f'learn sigma is {self.learn_sigma}')
        if self.learn_sigma:
            self._update_sigma()

            self.setup(stage="fit")

        self.train_queue = tio.Queue(
            self.train_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=self.num_workers,
        )
        # num_workers refers to the number of workers used to load and transform the volumes.
        # Multiprocessing is not needed to pop patches from the queue, so you should always use
        # num_workers=0 for the DataLoader you instantiate to generate training batches.
        return DataLoader(self.train_queue, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        # print('Creating val dataloader')
        if self.learn_sigma:
            self._update_sigma()

            self.setup(stage="fit")

        self.val_queue = tio.Queue(
            self.val_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=self.num_workers,
        )
        return DataLoader(self.val_queue, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        # print('creating test dataloader')
        if self.learn_sigma:
            self._update_sigma()

            self.setup(stage="test")

        self.test_queue = tio.Queue(
            self.test_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=self.num_workers,
        )
        return DataLoader(self.test_queue, batch_size=self.batch_size, num_workers=0)


class Model(pl.LightningModule):
    """Model class for the deep_radiologist network.

    Setup for use with pytorch Lightning.

    Args:
        config (dict): configuration dictionary (i.e. hyperparameters)
    """

    def __init__(self, config):
        super().__init__()

        print(f"Initiating model using the following config: {config}")

        self._model = UNet3D(
            in_channels=1,
            out_classes=1,
            num_encoding_blocks=config["num_encoding_blocks"],
            out_channels_first_layer=config["out_channels_first_layer"],
            normalization="batch",
            # pooling_type='max',
            pooling_type=config["pooling_type"],  # 'avg',
            # upsampling_type='conv',
            upsampling_type=config["upsampling_type"],  # 'linear',
            padding=True,
            activation=config["act"],
            dimensions=3,
            dropout=config["dropout"],
            output_activation=config["output_activation"],
        )
        # remove classifier from this U-Net implementation, as we are doing regression.
        self._model.classifier = torch.nn.Identity()

        self.criterion = torch.nn.MSELoss()
        self.optimizer_class = torch.optim.SGD

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.config = config

        self.debug_plots = config["debug_plots"]

        if config["visualise_model"]:
            print(self._model)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def prepare_batch(self, batch):
        # print('Make some logic here to concatenate the two types of labels into two channels')
        # breakpoint()
        # return batch['image'][tio.DATA], batch['label_corneas'][tio.DATA]
        return batch["image"][tio.DATA], batch["label"][tio.DATA]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.forward(x)

        if self.debug_plots:
            self.viewer = napari.view_image(x.cpu().numpy(), name="Input")
            self.viewer.add_image(y.cpu().numpy(), name="Ground Truth")
            self.viewer.add_image(y_hat.cpu().detach().numpy(), name="Prediction")

        return y_hat, y

    def forward(self, x):
        y_hat = self._model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=self.config["batch_size"]
        )

        # can remove to reduce cpu usage
        if self.debug_plots:
            tp, fp, fn, failures, mean_loc_err = self.calc_acc(y_hat, y)
        # self.log('train_tp', tp, prog_bar=True, batch_size=self.config['batch_size'])
        # self.log('train_fp', fp, prog_bar=True, batch_size=self.config['batch_size'])
        # self.log('train_fn', fn, prog_bar=True, batch_size=self.config['batch_size'])
        # self.log('train_failures', failures, prog_bar=True, batch_size=self.config['batch_size'])
        # self.log('train_mean_loc_err', mean_loc_err, prog_bar=True, batch_size=self.config['batch_size'])

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, batch_size=self.config["batch_size"])

        tp, fp, fn, failures, mean_loc_err = self.calc_acc(y_hat, y)
        self.log("val_tp", tp, batch_size=self.config["batch_size"])
        self.log("val_fp", fp, batch_size=self.config["batch_size"])
        self.log("val_fn", fn, batch_size=self.config["batch_size"])
        self.log("val_failures", failures, batch_size=self.config["batch_size"])
        self.log("val_mean_loc_err", mean_loc_err, batch_size=self.config["batch_size"])

        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, batch_size=self.config["batch_size"])

        return loss

    def _locate_coords(self, heatmap, min_distance=None, min_val=None):
        if min_distance is None:
            min_distance = self.config["peak_min_distance"]
        if min_val is None:
            min_val = self.config["peak_min_val"]

        coords = locate_peaks_in_volume(
            heatmap, min_distance=min_distance, min_val=min_val
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
                if j not in y_hat_match and y[j, i] not in y_match:
                    y_hat_match.append(j)
                    y_match.append(y[j, i])
                    dists.append(dists_k[j])

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
            try:
                loc_err = np.mean([np.mean(i) for i in loc_errs])
            except:
                import ipdb; ipdb.set_trace()
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
