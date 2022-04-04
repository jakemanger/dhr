from docstring_parser import compose
import pytorch_lightning as pl
import torchio as tio
import os
from torch.utils.data import random_split, DataLoader
import multiprocessing
import torch
import numpy as np
from scipy import spatial
import warnings

from mctnet.custom_unet import UNet3D
from mctnet.image_morph import crop_3d_coords
from mctnet.lazy_heatmap import LazyHeatmapReader
from mctnet.heatmap_peaker import locate_peaks_in_volume


class DataModule(pl.LightningDataModule):
    """ A pytorch lightning class to handle data loading and preprocessing

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
        self.train_images_dir = config['train_images_dir']
        self.train_labels_dir = config['train_labels_dir']
        self.test_images_dir = config['test_images_dir']
        self.test_labels_dir = config['test_labels_dir']
        self.histogram_landmarks_path = config['histogram_landmarks_path']
        self.image_suffix = config['image_suffix']
        self.label_suffix = config['label_suffix']
        self.train_val_ratio = config['train_val_ratio']
        self.batch_size = config['batch_size']
        self.patch_size = config['patch_size']
        self.samples_per_volume = config['samples_per_volume']
        self.max_length = config['max_length']
        self.balanced_sampler = config['balanced_sampler']
        self.balanced_sampler_max_length = config['balanced_sampler_max_length']
        self.ignore_empty_volumes = config['ignore_empty_volumes']
        self.sigma = config['starting_sigma']
        self.heatmap_max_length = config['heatmap_max_length']
        self.learn_sigma = config['learn_sigma']

        if config['num_workers'] in ('auto', 'Auto', 'AUTO'):
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = config['num_workers']

        if self.learn_sigma:
            raise NotImplementedError('Sigma learning not implemented yet')
        
        if not os.path.exists(self.histogram_landmarks_path):
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
        """ Finds the filenames of the images and labels in the given directories

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
            if file.endswith('.nii.gz'):
                spltstr = file.split('-')
                images.append(spltstr[0] + '-' + spltstr[1])
        for file in os.listdir(label_dir):
            if file.endswith('.csv') and self.label_suffix in file:
                if not self.ignore_empty_volumes or os.path.getsize(label_dir + file) > 0:
                    spltstr = file.split('-')
                    labels.append(spltstr[0] + '-' + spltstr[1])
            
        filenames = sorted(list(set(images) & set(labels)))
        print(f'Found {len(filenames)} labelled images with labels in {image_dir} and {label_dir} for analysis')
        return filenames

    def _load_subjects(self, image_dir, label_dir):
        """ Loads the images and labels from the given directories

        Args:
            image_dir (str): path to directory containing images
            label_dir (str): path to directory containing labels

        Returns:
            subjects (list): list of subjects (with images and labels)
        """

        subjects = []
        # find all the .nii files
        filenames = self._find_data_filenames(image_dir, label_dir)

        # now add them to a list of subjects
        for filename in filenames:
            nm_comps = filename.split('-')
            img = tio.ScalarImage(f'{image_dir}{nm_comps[0]}-{nm_comps[1]}-{self.image_suffix}.nii.gz', check_nans=True)

            heatmap_reader = LazyHeatmapReader(
                affine=img.affine,
                start_shape=img.shape,
                sigma=self.sigma,
                l=self.heatmap_max_length,
            )
            lbl=tio.Image(
                path=f'{label_dir}{nm_comps[0]}-{nm_comps[1]}-{self.label_suffix}.csv',
                type=tio.LABEL,
                check_nans=True,
                reader=heatmap_reader.read,
            )
            reader = LazyHeatmapReader(
                affine=img.affine,
                start_shape=img.shape,
                sigma=3,
                l=self.balanced_sampler_max_length,
                binary=True
            )
            smpl_map=tio.Image(
                path=f'{label_dir}{nm_comps[0]}-{nm_comps[1]}-{self.label_suffix}.csv',
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
                image=img,
                label=lbl,
                sampling_map=smpl_map,
                filename=filename
            )
            subjects.append(subject)
        return subjects

    def get_preprocessing_transform(self):
        """Returns the preprocessing transform for the dataset

        Returns:
            transform (torchvision.transforms.Compose): preprocessing transform
        """
        preprocess_list = [
            tio.ToCanonical(),
            tio.EnsureShapeMultiple(8)
        ]

        if self.config['histogram_standardisation']:
            preprocess_list.append(tio.HistogramStandardization(
                {'default_image_name': self.histogram_landmarks_path, f'{self.image_suffix}': self.histogram_landmarks_path},
                masking_method=tio.ZNormalization.mean
            ))
        
        if self.config['z_normalisation']:
            preprocess_list.append(tio.ZNormalization(masking_method=tio.ZNormalization.mean))

        return tio.Compose(preprocess_list)
    
    def get_augmentation_transform(self):
        """Returns the augmentation transform for the dataset

        Returns:
            transform (torchvision.transforms.Compose): augmentation transform
        """

        augment = tio.Compose([
            tio.RandomAffine(
                p=self.config['random_affine_prob'],
                scales=self.config['random_affine_scale_range'],
                degrees=self.config['random_affine_rotation_range'],
                translation=self.config['random_affine_translation_range'],
            ),
            tio.RandomElasticDeformation(
                p=self.config['random_elastic_deformation_prob'],
                num_control_points=self.config['random_elastic_deformation_num_control_points'],
                max_displacement=self.config['random_elastic_deformation_max_displacement'],
            ),
        ])
        return augment

    def get_sampler(self):
        if self.balanced_sampler:
            self.sampler = tio.LabelSampler(patch_size=self.patch_size, label_name='sampling_map', label_probabilities={0: 0.5, 1: 0.5})
        else:
            self.sampler = tio.UniformSampler(patch_size=self.patch_size)
    
    def create_histogram_landmarks(self):
        """Create histogram landmarks for the dataset.

        This is used to normalise the images. Is saved to the root directory.
        """

        print('Histogram landmarks not found, creating them...')
        
        # find all the .nii files
        filenames = self._find_data_filenames(self.train_images_dir, self.train_labels_dir)
        filenames = [self.train_images_dir + f + f'-{self.image_suffix}.nii.gz' for f in filenames]

        landmarks = tio.HistogramStandardization.train(
            filenames,
            output_path=self.histogram_landmarks_path,
            masking_function=tio.ZNormalization.mean
        )
        print(f'Histogram landmarks saved to {self.histogram_landmarks_path}')
        np.set_printoptions(suppress=True, precision=3)
        print('\nTrained landmarks:', landmarks)

    def setup(self, stage=None):
        """Sets up the dataset.

        Args:
            stage (str): stage to setup the dataset for (used internally by pytorch lightning)
                should be either 'fit' or 'test'
        """
        print('Running setup!')

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        if stage == 'fit' or stage is None:
            self.subjects = self._load_subjects(self.train_images_dir, self.train_labels_dir)
            num_subjects = len(self.subjects)
            num_train_subjects = int(round(num_subjects * self.train_val_ratio))
            num_val_subjects = num_subjects - num_train_subjects
            splits = num_train_subjects, num_val_subjects
            train_subjects, val_subjects = random_split(self.subjects, splits)
            self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
            self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        
        if stage == 'test' or stage is None:
            self.test_subjects = self._load_subjects(self.test_images_dir, self.test_labels_dir)
            self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

        self.get_sampler()

    def _update_sigma(self):
        raise NotImplementedError('Sigma learning not implemented yet')
        # self.sigma = self.trainer.model.sigma
        # self.sigma = 2
        # print(f'Sigma has been updated to {self.sigma}')

    def train_dataloader(self):
        # print('Creating train dataloader')
        # print(f'learn sigma is {self.learn_sigma}')
        if self.learn_sigma:
            self._update_sigma()

            self.setup(stage='fit')

        self.train_queue = tio.Queue(
            self.train_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=8,
        )
        # num_workers refers to the number of workers used to load and transform the volumes.
        # Multiprocessing is not needed to pop patches from the queue, so you should always use
        # num_workers=0 for the DataLoader you instantiate to generate training batches.
        return DataLoader(self.train_queue, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        # print('Creating val dataloader')
        if self.learn_sigma:
            self._update_sigma()

            self.setup(stage='fit')

        self.val_queue = tio.Queue(
            self.val_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=8,
        )
        return DataLoader(self.val_queue, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        # print('creating test dataloader')
        if self.learn_sigma:
            self._update_sigma()

            self.setup(stage='test')

        self.test_queue = tio.Queue(
            self.test_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=8
        )
        return DataLoader(self.test_queue, batch_size=self.batch_size, num_workers=0)



class Model(pl.LightningModule):
    """ Model class for the MCTNet network.

    Setup for use with pytorch Lightning.

    Args:
        config (dict): configuration dictionary (i.e. hyperparameters)
    """

    def __init__(self, config):
        super().__init__()

        print('Initiating model using the following config:')

        self._model = UNet3D(
            in_channels=1,
            out_classes=1,
            num_encoding_blocks=config['num_encoding_blocks'],
            out_channels_first_layer=config['out_channels_first_layer'],
            normalization='batch',
            # pooling_type='max',
            pooling_type=config['pooling_type'], #'avg',
            # upsampling_type='conv',
            upsampling_type=config['upsampling_type'], #'linear',
            padding=True,
            activation=config['act'],
            dimensions=3,
            dropout=config['dropout'],
            output_activation=config['output_activation'],
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer_class = torch.optim.SGD

        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        self.config = config

        self.debug_plots = config['debug_plots']

        if config['visualise_model']:
            print(self._model)

        self.save_hyperparameters()

    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def prepare_batch(self, batch):
        # print('Make some logic here to concatenate the two types of labels into two channels')
        # breakpoint()
        # return batch['image'][tio.DATA], batch['label_corneas'][tio.DATA]
        return batch['image'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.forward(x)

        if self.debug_plots:
            import napari
            viewer = napari.view_image(x.cpu().numpy(), name='Input')
            viewer.add_image(y.cpu().numpy(), name='Ground Truth')
            viewer.add_image(y_hat.cpu().detach().numpy(), name='Prediction')
            input('Press enter to continue...')
            viewer.close()

        return y_hat, y

    def forward(self, x):
        y_hat = self._model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)

        # can remove to reduce cpu usage
        # tp, fp, fn, failures, mean_loc_err = self.calc_acc(y_hat, y)
        # self.log('train_tp', tp, prog_bar=True)
        # self.log('train_fp', fp, prog_bar=True)
        # self.log('train_fn', fn, prog_bar=True)
        # self.log('train_failures', failures, prog_bar=True)
        # self.log('train_mean_loc_err', mean_loc_err, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

        tp, fp, fn, failures, mean_loc_err = self.calc_acc(y_hat, y)
        self.log('val_tp', tp)
        self.log('val_fp', fp)
        self.log('val_fn', fn)
        self.log('val_failures', failures)
        self.log('val_mean_loc_err', mean_loc_err)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

        return loss

    def _locate_coords(self, heatmap):
        coords = locate_peaks_in_volume(
            heatmap,
            min_distance=self.config['peak_min_distance'],
            min_val=self.config['peak_min_val']
        )
        return coords
    
    def _get_acc_metrics(self, y_hat, y):
        """Calculates accuracy metrics for a set of predicted and ground truth coordinates.

        Is a true positive if the distance between the predicted and closest ground truth coordinate
        is less than the correct_prediction_distance config parameter. Is a false positive if the 
        distance is greater than the correct_prediction_distance parameter or it already has a closer
        true positive. Is a false negative if the ground truth does not have a corresponding true
        positive.

        Args:
            y_hat (np.ndarray): predicted coordinates
            y (np.ndarray): ground truth coordinates

        Returns:
            tp (float): true positives
            fp (float): false positives
            fn (float): false negatives
            loc_errs (np.ndarray): location errors
        """

        tree = spatial.cKDTree(y)
        closest_dists, closest_nbrs = tree.query(y_hat, k=1)

        # if predictions are within distance of the same point, only keep the first one
        # this is to avoid repeated counting of true positives that are actually false positives
        # it doesn't matter which one is closer in this case, as we are just making a count
        removed_dup_indx = np.unique(closest_nbrs, return_index=True)[1]
        mask = np.zeros(closest_nbrs.shape, dtype='bool')
        mask[removed_dup_indx] = True

        true_positive = (closest_dists <= self.config['correct_prediction_distance']) & mask

        tp = len(true_positive[true_positive])
        fp = len(true_positive[~true_positive])
        fn = y.shape[0] - tp
        loc_errors = closest_dists[true_positive]

        if len(loc_errors) == 0:
            loc_errors = np.array([0])

        # import napari
        # tp_groundtruth = closest_nbrs[true_positive]
        # fn_mask = np.ones(y.shape[0], dtype='bool')
        # fn_mask[tp_groundtruth] = False
        # viewer = napari.view_points(y, name='all ground truth', size=2, face_color='pink')
        # viewer.add_points(y_hat[~true_positive], name='fp prediction', size=2, face_color='red')
        # viewer.add_points(y[fn_mask], name='fn', size=2, face_color='yellow')
        # viewer.add_points(y[closest_nbrs[true_positive]], name='tp groundtruth', size=2, face_color='blue')
        # viewer.add_points(y_hat[true_positive], name='tp prediction', size=2, face_color='green')
        # print(f'True positives: {tp}, False positives: {fp}, False negatives: {fn}, N Real values: {y.shape[0]}, N Predicted values: {y_hat.shape[0]}')
        # print(f'Mean Localisation error: {loc_errors.mean()}')
        # breakpoint()
        # viewer.close()

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

        for y_hat, y in zip(y_hats, ys):
            y_hat_coord = self._locate_coords(y_hat.cpu().detach().numpy())
            y_coord = self._locate_coords(y.cpu().detach().numpy())
            tp, fp, fn, loc_err = self._get_acc_metrics(y_hat_coord, y_coord)

            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            loc_errs.append(loc_err)

        tp = np.sum(tps)
        fp = np.sum(fps)
        fn = np.sum(fns)

        failures = fp + fn

        # import napari
        # viewr = napari.view_image(y.cpu().detach().numpy())
        # viewr.add_points(y_coord, size=2)
        # breakpoint()

        return (
            tp.astype(np.float32),
            fp.astype(np.float32),
            fn.astype(np.float32),
            failures.astype(np.float32),
            np.mean(np.concatenate(loc_errs)).astype(np.float32)
        )
