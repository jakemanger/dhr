from mctnet.lightning_modules import DataModule
import multiprocessing
import os
import torch

config = {
    'batch_size': 2,
    'patch_size': 64,
    'samples_per_volume': 40,
    'max_length': 400,
}

data = DataModule(
    batch_size=config['batch_size'],
    train_val_ratio=0.8,
    train_images_dir='./dataset/fiddler/images/',
    train_labels_dir='./dataset/fiddler/labels/',
    test_images_dir='./dataset/fiddler/test_images/',
    test_labels_dir='./dataset/fiddler/test_labels/',
    patch_size=config['patch_size'],
    samples_per_volume=config['samples_per_volume'],
    max_length=config['max_length'],
    num_workers=multiprocessing.cpu_count()
)

data.prepare_data()

subject_lists = [data.subjects, data.test_subjects]

inp = input(
    'Removing data without labels is an experimental idea.\n'
    'I am using this because becuase I have found that if there is a lot of empty space in the image,\n'
    'the model can easily fit to predict empty space and get stuck in a local minima.\n'
    'Are you sure you want to remove empty training data? (y/n)\n'
)

if inp != 'y' and inp != 'Y':
    print('Gettin\' outta here!')
    exit()

for subjects in subject_lists:
    for subject in subjects:
        unique_values = torch.tensor([])
        for key in tuple(subject.keys()):
            if type(subject[key]) != str and subject[key].type == 'label':
                unique_values = torch.cat((unique_values, subject[key].data.unique()), 0)

        if all(unique_values == 0):
            for key in tuple(subject.keys()):
                if type(subject[key]) != str:
                    file_to_remove = str(subject[key].path)
                    print(f'Removing empty data (scan {file_to_remove})')

                    # either choose to rename and ignore or delete the file
                    os.rename(file_to_remove, file_to_remove + '.empty')
                    # os.remove(file_to_remove)