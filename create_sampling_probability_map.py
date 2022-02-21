from mctnet.lightning_modules import DataModule
import multiprocessing
import os
import torch
import torchio as tio

config = {
    'batch_size': 2,
    'patch_size': 64,
    'samples_per_volume': 40,
    'max_length': 400,
}

data = DataModule(
    batch_size=config['batch_size'],
    train_val_ratio=0.8,
    train_images_dir='./dataset/fiddler/cropped/images/',
    train_labels_dir='./dataset/fiddler/cropped/labels/',
    test_images_dir='./dataset/fiddler/cropped/test_images/',
    test_labels_dir='./dataset/fiddler/cropped/test_labels/',
    patch_size=config['patch_size'],
    samples_per_volume=config['samples_per_volume'],
    max_length=config['max_length'],
    num_workers=multiprocessing.cpu_count()
)

data.prepare_data()

subject_lists = [data.subjects, data.test_subjects]


for subjects in subject_lists:
    for subject in subjects:
        unique_values = torch.tensor([])

        path = subject['label'].path
        print(f'Creating sampling map (scan {str(path)})')

        img = tio.ScalarImage(path)

        # make image binary
        img.set_data(img.data > 0)

        names = str(path).split('.')
        if len(names) > 2:
            img.save(f'{names[0]}.sampling_map.{names[1]}.{names[2]}')
        else:
            img.save(f'{names[0]}.sampling_map.{names[1]}')
