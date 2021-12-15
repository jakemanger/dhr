import random
import multiprocessing
from datetime import datetime
import torch
import torchio as tio
import pytorch_lightning as pl
import torch.nn
import monai
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import napari
import sys
from tqdm import tqdm
from pathlib import Path

from mctnet.lightning_modules import DataModule, Model

# to monitor training, run this in terminal:
# tensorboard --logdir lightning_logs

# TODO:
# implement hyperparameter tunings using ray tune
# add support to ensure that each image has a label for it to be sampled (and make this a hyperparameter option to learn)
# see if I can quickly generate a sigma value for the gaussian noise around labels like Payer et al.
# allow the test sampler the ability to sample with a grid
# allow the outputs to be reconstructed, if using a grid sampler

# hyperparameters taken from https://link.springer.com/chapter/10.1007/978-3-319-46723-8_27#CR12
config = {
    'lr': 1e-2,
    'weight_decay': 0.,
    'momentum': 0.99,
    'batch_size': 2,
    'patch_size': 64,
    'samples_per_volume': 40,
    'max_length': 400,
    'features': (64, 64, 128, 256, 512, 64),
    'act': 'relu'
}

seed = 42

device = torch.device('cuda')

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

def setup():
    random.seed(seed)
    torch.manual_seed(seed)
    plt.rcParams['figure.figsize'] = 12, 6
    print('TorchIO version:', tio.__version__)

    data.prepare_data()
    data.setup()
    print('Training:  ', len(data.train_set))
    print('Validation: ', len(data.val_set))
    print('Test:      ', len(data.test_set))

    # TODO: check what 16bit precision is

    return data


def train():
    """
    Trains the model using hyperparameters from config (at top of script).
    """

    data = setup()

    model = Model(
        config=config
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        callbacks=[checkpoint_callback],
        max_epochs=2000000,
    )
    trainer.logger._default_hp_metric = False

    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)


def inference(checkpoint_path, volume_path, aggregate_and_save=True, patch_size=128, patch_overlap=32, batch_size=1):
    """
    Produces a plot of the model's predictions on the test set.
    """

    model = Model.load_from_checkpoint(checkpoint_path).to(device)

    if aggregate_and_save:
        preprocess = data.get_preprocessing_transform()
        subjects = [
            tio.Subject(
                image=tio.ScalarImage(volume_path, check_nans=True),
            )
        ]
        subjects = tio.SubjectsDataset(subjects, transform=preprocess)

        print('Creating sampler and applying transform to image...')
        grid_sampler = tio.inference.GridSampler(
            subjects[0],
            patch_size,
            patch_overlap
        )

        print('Initialising patch_loader and aggregator...')
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

        print('Starting inference...')
        model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                x = patches_batch['image'][tio.DATA].to(device)
                locations = patches_batch[tio.LOCATION]
                y_hat = model(x)
                aggregator.add_batch(y_hat, locations)

            prediction = tio.Image(tensor=aggregator.get_output_tensor(), type=tio.LABEL)
            prediction_path = Path(volume_path).with_suffix('.prediction.nii.gz')
            prediction.save(prediction_path)
    else:
        model.eval()
        with torch.no_grad():
            viewer = napari.Viewer(title='Inputs, Labels and Predictions', ndisplay=3)

            for batch in data.test_dataloader():
            # for batch in data.val_dataloader():
                inputs = batch['image'][tio.DATA].to(model.device)
                y = batch['label_corneas'][tio.DATA].to(model.device)
                pred_y = model(inputs).to(device)

                # plot
                for i in range(len(inputs)): # loop though each volume in batch and plot
                    viewer.add_image(inputs[i, ...].cpu().numpy(), name='x', contrast_limits=(0, 1))
                    viewer.add_image(y[i, ...].cpu().numpy(), name='y', contrast_limits=(0, 1))
                    viewer.add_image(pred_y[i, ...].cpu().numpy(), name='y_hat', contrast_limits=(0, 1))
                    input('Press enter to continue')
                    viewer.layers.clear()


if __name__ == '__main__':
    USAGE = (
        '''
        Usage:
        To train a new model and generate a checkpoint with model parameters:
        python main.py [train]
        or
        To run inference on a volume using the default checkpoint with model parameters:
        python main.py [inference] [volume_path]
        or
        To run inference on a volume using the a specific checkpoint with model parameters:
        python main.py [inference] [volume_path] [checkpoint_path]   
        '''
    )
    args = sys.argv[1:]

    if not args or args[0] not in ['train', 'inference']:
        raise SystemExit(USAGE)

    if args[0] == 'inference':
        if len(args) < 2:
            print('No volume_path argument found')
            raise SystemExit(USAGE)

        if len(args) < 3:
            print('No checkpoint argument found, loading default checkpoint')
            args.append('lightning_logs/version_31/checkpoints/epoch=181-step=706159.ckpt')

    if args[0] == 'train':
        train()
    elif args[0] == 'inference':
        inference(args[1], args[2])
