import os
import gc
import napari
import torch.nn
import matplotlib.pyplot as plt
import random
import multiprocessing
from datetime import datetime
import torch
import torchio as tio
import pytorch_lightning as pl
from tqdm import tqdm
from pathlib import Path
import numpy as np

from optuna.integration import PyTorchLightningPruningCallback
import optuna

from mctnet.lightning_modules import DataModule, Model
from mctnet.heatmap_peaker import locate_peaks_in_volume


device = torch.device('cuda')

def init_data(config, run_internal_setup_func=False):
    """Initializes the data module.

    run_internal_setup_func should be set to True if you are not using
    pytorch lightning's `Trainer` to train the model (e.g. during inference),
    as `Trainer` will call data.setup() for you.

    Args:
        config (dict): The config dictionary.
        run_internal_setup_func (bool): Whether to run the internal setup function.

    Returns:
        DataModule: The data module.
    """
    data = DataModule(
        batch_size=config['batch_size'],
        train_val_ratio=config['train_val_ratio'],
        train_images_dir=config['train_images_dir'],
        train_labels_dir=config['train_labels_dir'],
        test_images_dir=config['test_images_dir'],
        test_labels_dir=config['test_labels_dir'],
        patch_size=config['patch_size'],
        samples_per_volume=config['samples_per_volume'],
        max_length=config['max_length'],
        num_workers=multiprocessing.cpu_count()
    )

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    plt.rcParams['figure.figsize'] = 12, 6
    print('TorchIO version:', tio.__version__)

    if run_internal_setup_func:
        data.setup()
        print('Training:  ', len(data.train_set))
        print('Validation: ', len(data.val_set))
        print('Test:      ', len(data.test_set))

    return data


def train(config, num_epochs=200, show_progress=False):
    """Trains the model using hyperparameters from config (at top of script).

    Args:
        config (dict): The config dictionary.
        num_epochs (int): The number of epochs to train for.
        show_progress (bool): Whether to show a progress bar.
    """

    data = init_data(config)

    model = Model(
        config=config
    )

    if show_progress:
        progress_bar_refresh_rate = 1
    else:
        progress_bar_refresh_rate = 0

    # check for no improvement over 5 epochs
    # and end early if so
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    # save a model checkpoint every 20 epochs
    # also save top 3 models with minimum validation loss
    # and the last model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True
    )
    every_n_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        every_n_epochs=20
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        callbacks=[early_stopping, checkpoint_callback, every_n_checkpoint_callback],
        max_epochs=num_epochs,
        progress_bar_refresh_rate=progress_bar_refresh_rate
    )
    trainer.logger._default_hp_metric = False

    start = datetime.now()
    print('Training started at', start)

    trainer.fit(model=model, datamodule=data)

    print('Training duration:', datetime.now() - start)


def objective(trial: optuna.trial.Trial, config, num_epochs, show_progress=True):
    """ Objective function for optuna.

    Args:
        trial (optuna.trial.Trial): The trial object.
        config (dict): The config dictionary.
        num_epochs (int): The number of epochs to train for.
        show_progress (bool): Whether to show progress.
    
    Returns:
        float: The validation loss.

     Should be called with
     ```
     study.optimize(lambda trial: objective(trial, config, num_epochs), n_trials=50)
     ```
    """

    # set possible hyperparameters to tune
    config['lr'] = trial.suggest_loguniform('lr', 1e-10, 1e-1)
    config['weight_decay'] = trial.suggest_categorical('weight_decay', [0, 1e-2, 1e-4, 1e-6])
    config['momentum'] = trial.suggest_uniform('momentum', 0.9, 0.99)
    config['batch_size'] = trial.suggest_categorical('batch_size', [1, 2])
    config['patch_size'] = trial.suggest_categorical('patch_size', [32, 64])
    config['features_scalar'] = trial.suggest_categorical('features_scalar', [0.5, 1])

    
    if show_progress:
        progress_bar_refresh_rate = 1
    else:
        progress_bar_refresh_rate = 0

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join('lightning_logs', trial.study.study_name, f'trial_{trial.number}'),
        monitor="val_loss",
    )
    pruning_callback = PyTorchLightningPruningCallback(trial, 'val_loss')
    # check for no improvement over 5 epochs
    # and end early if so
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    model = Model(
        config=config
    )
    data = init_data(config)

    trainer = pl.Trainer(
        logger = True,
        gpus=1 if torch.cuda.is_available() else None,
        precision=16,
        callbacks=[checkpoint_callback, pruning_callback, early_stopping],
        max_epochs=num_epochs,
        progress_bar_refresh_rate=progress_bar_refresh_rate
    )

    trainer.logger.log_hyperparams(config)

    trainer.fit(model=model, datamodule=data)

    # potential fix for deadlock issue
    # kill all the workers when the trial has finished
    del(data.train_queue._subjects_iterable)
    del(data.train_queue)
    del(data.val_queue._subjects_iterable)
    del(data.val_queue)
    del(data.test_queue._subjects_iterable)
    del(data.test_queue)
    del(data)

    gc.collect()
    
    return trainer.callback_metrics['val_loss'].item()


def inference(config, checkpoint_path, volume_path, aggregate_and_save=True, patch_size=128, patch_overlap=32, batch_size=1, transform_patch=True):
    """Produces a plot of the model's predictions on the test set.

    Args:
        config (dict): The configuration dictionary.
        checkpoint_path (str): The path to the model checkpoint.
        volume_path (str): The path to the volume to be predicted.
        aggregate_and_save (bool): Whether to aggregate the predictions and save them to a file. If false, assumes you are
            running inference on the test set of small cropped images.
        patch_size (int): The size of the patches to be used for inference.
        patch_overlap (int): The amount of overlap between the patches.
        batch_size (int): The batch size to use for inference.
        transform_patch (bool): Whether to transform each patch.
            This is slower, but uses much less RAM if the input volume is large (e.g. > 1GB). If false,
            the whole volume is loaded into RAM and transformed at the start.
    """
    data = init_data(config, run_internal_setup_func=True)

    model = Model.load_from_checkpoint(checkpoint_path).to(device)

    if aggregate_and_save:
        preprocess = data.get_preprocessing_transform()
        subjects = [
            tio.Subject(
                image=tio.ScalarImage(volume_path, check_nans=True),
            )
        ]
        # apply transform to whole image
        if transform_patch:
            print('Creating sampler...')
            subjects = tio.SubjectsDataset(subjects)
        else:
            print('Creating sampler and applying transform to image...')
            subjects = tio.SubjectsDataset(subjects, transform=preprocess)

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
                if transform_patch:
                    # apply transform to each patch individually
                    x = patches_batch['image'][tio.DATA].to(device)

                    # if there are only 0 values, normalization will fail
                    # so this is a workaround
                    if x.sum() == 0:
                        temp_preprocess = tio.Compose([preprocess[0], preprocess[3]])
                        x = temp_preprocess(x[0]).float()
                    else:
                        temp_preprocess = preprocess
                        x = temp_preprocess(x[0].to('cpu'))

                    x = x.unsqueeze(0).to(device)
                else:
                    # transform was already applied
                    x = patches_batch['image'][tio.DATA].to(device)

                locations = patches_batch[tio.LOCATION]
                y_hat = model(x)
                aggregator.add_batch(y_hat, locations)

            prediction = tio.Image(tensor=aggregator.get_output_tensor(), type=tio.LABEL)
            prediction_path = Path(volume_path).with_suffix('.prediction.nii.gz')

            prediction.save(prediction_path)

            breakpoint()

            locate_peaks(prediction, save=True)

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


def locate_peaks(heatmap, save=True, plot=True):
    if type(heatmap) == str:
        heatmap = tio.Image(heatmap, type=tio.LABEL)

    if plot:
        print('Plotting heatmap...')
        viewer = napari.view_image(heatmap.numpy(), name='heatmap')

    print('Locating peaks...')
    peaks = locate_peaks_in_volume(heatmap.numpy(), min_distance=4, min_val=0.2)

    if save:
        print('Saving peaks...')
        peaks_path = Path(heatmap.path).with_suffix('.peaks.csv')
        np.savetxt(peaks_path, peaks, delimiter=',')

    if plot:
        print('Plotting peaks...')
        viewer.add_points(peaks, name='peaks')
        breakpoint()

    return peaks
