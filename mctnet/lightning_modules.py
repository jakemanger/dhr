import pytorch_lightning as pl
import torchio as tio
import os
from torch.utils.data import random_split, DataLoader
import monai
import torch
import numpy as np

# from multiprocessing import Manager
# class SubjectsDataset(tio.SubjectsDataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Only changes.
#         manager = Manager()
#         self._subjects = manager.list(self._subjects)


class DataModule(pl.LightningDataModule):
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
        num_workers = 4,
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

    def get_max_shape(self, subjects):
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def _find_data_filenames(self, image_dir, label_dir):
        images = []
        labels = []
        for file in os.listdir(image_dir):
            if file.endswith('.nii.gz'):
                spltstr = file.split('-')
                images.append(spltstr[0] + '-' + spltstr[1])
        for file in os.listdir(label_dir):
            if file.endswith('.nii.gz'):
                spltstr = file.split('-')
                labels.append(spltstr[0] + '-' + spltstr[1])
            
        filenames = sorted(list(set(images) & set(labels)))
        print(f'Found {len(filenames)} labelled images with labels in {image_dir} and {label_dir} for analysis')
        return filenames

    def _load_subjects(self, image_dir, label_dir):
        subjects = []
        # find all the .nii files
        filenames = self._find_data_filenames(image_dir, label_dir)
        # now add them to a list of subjects
        for filename in filenames:
            nm_comps = filename.split('-')
            subject = tio.Subject(
                image=tio.ScalarImage(f'{image_dir}{nm_comps[0]}-{nm_comps[1]}-image.nii.gz', check_nans=True),
                label_corneas=tio.Image(f'{label_dir}{nm_comps[0]}-{nm_comps[1]}-corneas.nii.gz', type=tio.LABEL, check_nans=True),
                label_rhabdoms=tio.Image(f'{label_dir}{nm_comps[0]}-{nm_comps[1]}-rhabdoms.nii.gz', type=tio.LABEL, check_nans=True),
                filename=filename
            )
            subjects.append(subject)
        return subjects

    def prepare_data(self):
        # get train/val (subjects) and test subjects (test_subjects)
        self.subjects = self._load_subjects(self.train_images_dir, self.train_labels_dir)
        self.test_subjects = self._load_subjects(self.test_images_dir, self.test_labels_dir)
        
    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.ToCanonical(),
            tio.HistogramStandardization({'image': '/home/jake/projects/mctnet/landmarks.npy'}, masking_method=tio.ZNormalization.mean),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.EnsureShapeMultiple(8) # for the u-net TODO check if this needs updating as I have changed my model features,
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomMotion(p=0.2),
            tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            tio.RandomAffine(p=0.8)
            # tio.OneOf({
            #     tio.RandomAffine(): 0.8,
            #     tio.RandomElasticDeformation(
            #         max_displacement=[10, 10, 10]
            #     ): 0.2,
            # }, p=0.8),
        ])
        return augment

    def get_sampler(self):
        self.sampler = tio.UniformSampler(patch_size=self.patch_size)

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

        self.get_sampler()

        self.train_queue = tio.Queue(
            self.train_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=self.num_workers
        )
        self.val_queue = tio.Queue(
            self.val_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=self.num_workers
        )
        self.test_queue = tio.Queue(
            self.test_set,
            self.max_length,
            self.samples_per_volume,
            self.sampler,
            num_workers=self.num_workers
        )

    
    def train_dataloader(self):
        # num_workers refers to the number of workers used to load and transform the volumes.
        # Multiprocessing is not needed to pop patches from the queue, so you should always use
        # num_workers=0 for the DataLoader you instantiate to generate training batches.
        return DataLoader(self.train_queue, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_queue, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_queue, batch_size=self.batch_size, num_workers=0)


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self._model = monai.networks.nets.BasicUnet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            features=config['features'],
            act=config['act']
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer_class = torch.optim.SGD

        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def prepare_batch(self, batch):
        # print('Make some logic here to concatenate the two types of labels into two channels')
        # breakpoint()
        return batch['image'][tio.DATA], batch['label_corneas'][tio.DATA]
    
    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.forward(x)
        return y_hat, y

    def forward(self, x):
        return self._model(x)

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