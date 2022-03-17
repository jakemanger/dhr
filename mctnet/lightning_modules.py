import pytorch_lightning as pl
import torchio as tio
import os
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np

from mctnet.custom_unet import UNet3D
from mctnet.lazy_heatmap import LazyHeatmapReader


class DataModule(pl.LightningDataModule):
    """ A pytorch lightning class to handle data loading and preprocessing

    Uses the torchio library to load and preprocess data and
    returns a dataloader for training and validation.

    Args:
        batch_size (int): batch size for training and validation
        train_val_ratio (float): ratio of training data to validation data
        train_images_dir (str): path to directory containing training images
        train_labels_dir (str): path to directory containing training labels
        test_images_dir (str): path to directory containing test images
        test_labels_dir (str): path to directory containing test labels
        patch_size (tuple): size of patches to extract from images
        samples_per_volume (int): number of patches to extract from each volume
        max_length (int): maximum length of patches to extract from images
        num_workers (int): number of workers to use for data loading
        balanced_sampler (bool): whether to use a balanced sampler
        image_suffix (str): suffix of the image files
        label_suffix (str): suffix of the label files
        sigma (float): sigma to use for heatmap generation
        learn_sigma (bool): whether to learn sigma
        heatmap_max_length (int): maximum length of heatmaps to generate
        balanced_sampler_length (int): length of balanced sampler
        ignore_empty_volumes (bool): whether to ignore volumes with no labels
        histogram_landmarks_path (str): path to histogram landmarks file for histogram standardization. If it does not exist, it will be created using training/val (but NOT test) data
    """

    def __init__(
        self,
        batch_size,
        train_val_ratio,
        train_images_dir,
        train_labels_dir,
        test_images_dir,
        test_labels_dir,
        patch_size,
        samples_per_volume,
        max_length, 
        num_workers = 15,
        balanced_sampler = True,
        image_suffix = 'image',
        label_suffix = 'corneas',
        sigma = 2,
        learn_sigma = False,
        heatmap_max_length = 25,
        balanced_sampler_length = 9,
        ignore_empty_volumes = True,
        histogram_landmarks_path = './fiddlercrab_landmarks.npy',
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.test_images_dir = test_images_dir
        self.test_labels_dir = test_labels_dir
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.max_length = max_length
        self.num_workers = num_workers
        self.balanced_sampler = balanced_sampler
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.sigma = sigma
        self.learn_sigma = learn_sigma
        self.heatmap_max_length = heatmap_max_length
        self.balanced_sampler_length = balanced_sampler_length
        self.ignore_empty_volumes = ignore_empty_volumes
        self.histogram_landmarks_path = histogram_landmarks_path

        if learn_sigma:
            raise NotImplementedError('Sigma learning not implemented yet')
        
        if not os.path.exists(histogram_landmarks_path):
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
                l=self.balanced_sampler_length,
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

        preprocess = tio.Compose([
            tio.ToCanonical(),
            tio.HistogramStandardization(
                {'default_image_name': self.histogram_landmarks_path, f'{self.image_suffix}': self.histogram_landmarks_path},
                masking_method=tio.ZNormalization.mean
            ),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.EnsureShapeMultiple(8) # for the u-net TODO check if this needs updating as I have changed my model features,
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        """Returns the augmentation transform for the dataset

        Returns:
            transform (torchvision.transforms.Compose): augmentation transform
        """

        augment = tio.Compose([
            tio.RandomAffine(scales=0.3, degrees=360, p=1),
            tio.RandomElasticDeformation(p=0.25),
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
        self.sigma = 2
        print(f'Sigma has been updated to {self.sigma}')

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
        print(config)

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
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss
