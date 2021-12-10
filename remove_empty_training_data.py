from mctnet.lightning_modules import DataModule
import multiprocessing
import os

config = {
    'batch_size': 2,
    'patch_size': 64,
    'samples_per_volume': 40,
    'max_length': 400,
}

data = DataModule(
    batch_size=config['batch_size'],
    train_val_ratio=0.8,
    train_images_dir='./dataset/crab_images/',
    train_labels_dir='./dataset/crab_labels/',
    test_images_dir='./dataset/crab_test_images/',
    test_labels_dir='./dataset/crab_test_labels/',
    patch_size=config['patch_size'],
    samples_per_volume=config['samples_per_volume'],
    max_length=config['max_length'],
    num_workers=multiprocessing.cpu_count()
)

data.prepare_data()

subject_lists = [data.subjects, data.test_subjects]

for subjects in subject_lists:
    for subject in subjects:
        if (
            all(subject['label_corneas'].data.unique() == 0)
        ):
            file_to_remove = str(subject['label_rhabdoms'].path)
            print(f'Removing empty data (scan {file_to_remove})')
            os.remove(file_to_remove)

            file_to_remove = str(subject['label_corneas'].path)
            print(f'Removing empty data (scan {file_to_remove})')
            os.remove(file_to_remove)

            file_to_remove = str(subject['image'].path)
            print(f'Removing empty data (scan {file_to_remove})')
            os.remove(file_to_remove)