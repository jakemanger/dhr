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
from ray.tune.integration.pytorch_lightning import TuneReportCallback

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
    """
    Trains the model using hyperparameters from config (at top of script).
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
    # report to ray tune on validation end
    metrics = {"loss": "val_loss"}
    hyperparam_tune_callback = TuneReportCallback(metrics, on='validation_end')

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        callbacks=[early_stopping, checkpoint_callback, every_n_checkpoint_callback, hyperparam_tune_callback],
        max_epochs=num_epochs,
        progress_bar_refresh_rate=progress_bar_refresh_rate
    )
    trainer.logger._default_hp_metric = False

    start = datetime.now()
    print('Training started at', start)

    try:    
        trainer.fit(model=model, datamodule=data)
    finally:
        print('I was killed. Deleting processes and exiting...')
        del(data)
        del(model)
        del(trainer)
        torch.cuda.empty_cache()

    print('Training duration:', datetime.now() - start)


def inference(config, checkpoint_path, volume_path, aggregate_and_save=True, patch_size=128, patch_overlap=32, batch_size=1):
    """
    Produces a plot of the model's predictions on the test set.
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

            prediction_path = Path(volume_path).with_suffix('.prediction_coords.csv')
            prediction.save(prediction_path)

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
        heatmap = tio.Image(heatmap)

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

    return peaks