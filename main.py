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


def setup():
    random.seed(seed)
    torch.manual_seed(seed)
    plt.rcParams['figure.figsize'] = 12, 6
    print('TorchIO version:', tio.__version__)

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


def inference(napari_plot=True):
    """
    Produces a plot of the model's predictions on the test set.
    """
    data = setup()

    model = Model.load_from_checkpoint(
        checkpoint_path='lightning_logs/version_31/checkpoints/epoch=181-step=706159.ckpt',
        hparams_file='lightning_logs/version_31/hparams.yaml',
    )

    with torch.no_grad():
        if napari_plot:
            viewer = napari.Viewer(title='Inputs, Labels and Predictions', ndisplay=3)

        for batch in data.test_dataloader():
        # for batch in data.val_dataloader():
            inputs = batch['image'][tio.DATA].to(model.device)
            y = batch['label_corneas'][tio.DATA].to(model.device)
            pred_y = model(inputs)

            # plot
            if napari_plot:
                for i in range(len(inputs)): # loop though each volume in batch and plot
                    viewer.add_image(inputs[i, ...].cpu().numpy(), name='input', contrast_limits=(0, 1))
                    viewer.add_image(y[i, ...].cpu().numpy(), name='y', contrast_limits=(0, 1))
                    viewer.add_image(pred_y[i, ...].cpu().numpy(), name='pred_y', contrast_limits=(0, 1))
                    input('Press enter to continue')
                    viewer.layers.clear()
            else:
                _, axes = plt.subplots(3, len(inputs))
                for i in range(len(inputs)):
                    axes[0, i].imshow(inputs[i, 0, 128, :, :].cpu().numpy(), cmap='gray')
                    axes[0, i].set_title('Input')
                    axes[1, i].imshow(y[i, 0, 128, :, :].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title('Ground truth')
                    axes[2, i].imshow(pred_y[i, 0, 128, :, :].cpu().numpy(), cmap='gray')
                    axes[2, i].set_title('Output')
                    input('Press enter to continue')


if __name__ == '__main__':
    USAGE = 'Please specify a command: train or inference. E.g. python main.py train'
    args = sys.argv[1:]

    if not args or args[0] not in ['train', 'inference']:
        raise SystemExit(USAGE)

    if args[0] == 'train':
        train()
    elif args[0] == 'inference':
        inference()