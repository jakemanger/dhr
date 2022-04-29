from deep_radiologist.lightning_modules import DataModule
import multiprocessing
import os
import torch
import argparse
import yaml
from yaml.loader import SafeLoader


def main():

    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument(
        'config_path',
        type=str,
        help='''
        The path to the config file.

        Example:
            configs/fiddlercrab_cornea_config.yaml
        '''
    )

    args = parser.parse_args()

    # load config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)

    data = DataModule(
        batch_size=config['batch_size'],
        train_val_ratio=0.8,
        train_images_dir=config['train_images_dir'],
        train_labels_dir=config['train_labels_dir'],
        test_images_dir=config['test_images_dir'],
        test_labels_dir=config['test_labels_dir'],
        patch_size=config['patch_size'],
        samples_per_volume=config['samples_per_volume'],
        max_length=config['max_length'],
        num_workers=multiprocessing.cpu_count()
    )

    data.prepare_data()

    subject_lists = [data.subjects, data.test_subjects]

    inp = input(
        'Marking empty training data to be ignored.\n'
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


if __name__ == '__main__':
    main()
