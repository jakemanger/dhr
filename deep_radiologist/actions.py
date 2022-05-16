import os
import gc
import napari
import torch.nn
import matplotlib.pyplot as plt
import random
from datetime import datetime
import torch
import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
from pathlib import Path
import numpy as np
import yaml

from optuna.integration import PyTorchLightningPruningCallback
import optuna

from deep_radiologist.inference_manager import InferenceManager
from deep_radiologist.lightning_modules import DataModule, Model
from deep_radiologist.heatmap_peaker import locate_peaks_in_volume


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
        config=config
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


def train(config, num_epochs=1500, show_progress=False):
    """Trains the model using hyperparameters from config

    Args:
        config (dict): The config dictionary.
        num_epochs (int): The maximum number of epochs to train for if early stopping doesn't
            occur.
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

    # check for no improvement over 10 epochs
    # and end early if so.
    # this is to prevent overfitting to training data (but not
    # validation data), or if has better loss at the cost of
    # our accuracy metric ('val_failures')
    # early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    #     monitor='val_failures',
    #     patience=10
    # )
    # save a model checkpoint every 20 epochs
    # also save top 3 models with minimum validation loss
    # and the last model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        every_n_epochs=20,
    )

    save_path=os.path.join('logs', config['config_stem'])

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        # callbacks=[early_stopping, checkpoint_callback],
        callbacks=[checkpoint_callback],
        max_epochs=num_epochs,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        reload_dataloaders_every_epoch=True if config['learn_sigma'] else False,
        enable_checkpointing=True,
        default_root_dir=save_path,
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

    # var_to_optimise = 'val_loss'
    var_to_optimise = 'val_failures'

    # set possible hyperparameters to tune
    config['lr'] = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    config['weight_decay'] = trial.suggest_categorical('weight_decay', [0, 1e-1])
    config['momentum'] = trial.suggest_uniform('momentum', 0.9, 0.99)
    config['num_encoding_blocks'] = trial.suggest_categorical('num_encoding_blocks', [3, 4, 5])
    config['out_channels_first_layer'] = trial.suggest_categorical('out_channels_first_layer', [32, 64])
    # config['pooling_type'] = trial.suggest_categorical('pooling_type', ['max', 'avg'])
    # config['upsampling_type'] = trial.suggest_categorical('upsampling_type', ['linear', 'conv'])
    # config['act'] = trial.suggest_categorical('act', ['ReLU', 'LeakyReLU'])
    config['dropout'] = trial.suggest_categorical('dropout', [0, 0.1])
    config['starting_sigma'] = trial.suggest_uniform('starting_sigma', 1, 4) 
    config['random_affine_prob'] = trial.suggest_uniform('random_affine_prob', 0.0, 1.0)
    config['random_elastic_deformation_prob'] = trial.suggest_uniform('random_elastic_deformation_prob', 0.0, 0.3)
    config['histogram_standardisation'] = trial.suggest_categorical('histogram_standardisation', [True, False])
    

    if show_progress:
        progress_bar_refresh_rate = 1
    else:
        progress_bar_refresh_rate = 0

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
    )
    pruning_callback = PyTorchLightningPruningCallback(trial, var_to_optimise)
    model = Model(
        config=config
    )
    data = init_data(config)

    save_path = os.path.join('logs', config['config_stem'], 'hyperparameter_tuning', f'version_{trial.number}')

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        precision=16,
        callbacks=[checkpoint_callback, pruning_callback],
        max_epochs=num_epochs,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        reload_dataloaders_every_epoch=True if config['learn_sigma'] else False,
        enable_checkpointing=True,
        default_root_dir=save_path
    )

    trainer.logger.log_hyperparams(config)

    trainer.fit(model=model, datamodule=data)

    # fix for deadlock issue
    # kill all the workers when the trial has finished
    if hasattr(data, 'train_queue'):
        del(data.train_queue._subjects_iterable)
        del(data.train_queue)
    if hasattr(data, 'val_queue'):
        del(data.val_queue._subjects_iterable)
        del(data.val_queue)
    if hasattr(data, 'test_queue'):
        del(data.test_queue._subjects_iterable)
        del(data.test_queue)
    if data in globals():
        del(data)

    gc.collect()
    
    return trainer.callback_metrics[var_to_optimise].item()


def inference(
    config_path,
    checkpoint_path,
    volume_path,
    aggregate_and_save=True,
    patch_size=128,
    patch_overlap=32,
    # patch_overlap=0,
    batch_size=3,
    n_x_dirs=2,
    n_y_dirs=1,
    n_z_dirs=2,
    debug_patch_plots=False,
    debug_volume_plots=False
):
    """Produces a plot of the model's predictions on the test set.

    Args:
        config_path (dict): Path to a hparams .yaml file with a configuration dictionary.
        checkpoint_path (str): The path to the model checkpoint.
        volume_path (str): The path to the volume to be predicted.
        aggregate_and_save (bool): Whether to aggregate the predictions of the whole volume and save them to a file.
            If false, assumes you are running inference on the test set of small cropped images.
        patch_size (int): The size of the patches to be used for inference.
        patch_overlap (int): The amount of overlap between the patches.
        batch_size (int): The batch size to use for inference.
        n_x_dirs (int): The number of x-directions to use for inference. Automatically evenly spreads these from 0-180°.
        n_y_dirs (int): The number of y-directions to use for inference. Automatically evenly spreads these from 0-180°.
        n_z_dirs (int): The number of z-directions to use for inference. Automatically evenly spreads these from 0-180°.
        debug_patch_plots (bool): Whether to show debug plots of inference on a single patch. Shows this in different rotations
            along the z-axis.
        debug_volume_plots (bool): Whether to show debug plots of inference on the whole volume. Shows this in different rotations
    
    Returns:
        If aggregate_and_save is true, returns the path to the aggregated predictions. Otherwise, returns None.
    """
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)['config']
    data = init_data(config, run_internal_setup_func=True)

    model = Model.load_from_checkpoint(checkpoint_path, hparams_file=config_path).to(device)

    if aggregate_and_save:
        x_rotations = range(0, 180, 180 // n_x_dirs)
        y_rotations = range(0, 180, 180 // n_y_dirs)
        z_rotations = range(0, 180, 180 // n_z_dirs)

        im = InferenceManager(
            volume_path,
            model,
            data,
            patch_size,
            patch_overlap,
            batch_size,
        )

        prediction = im.run(
            x_rotations,
            y_rotations,
            z_rotations,
            debug_patch_plots,
            debug_volume_plots,
            combination_mode = 'average'
        )

        prediction_path = (
            './output/'
            + str(Path(Path(volume_path).stem).with_suffix(
                    '.'
                    + os.path.relpath(checkpoint_path).replace('/', '_').replace('.ckpt', '_')
                    + 'prediction.nii.gz'
                )
            )
        )

        print(f'Saving prediction to {prediction_path}')
        prediction.save(prediction_path)

        return prediction_path

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


def locate_peaks(heatmap_path, save=True, plot=True, peak_min_dist=5, peak_min_val=0.4):
    """Locate the peaks in a heatmap.

    Args:
        heatmap_path (str): The path to the heatmap to be processed.
        save (bool): Whether to save the results.
        plot (bool): Whether to plot the results.
        peak_min_dist (int): The minimum distance between peaks used when calculating coordinates of object locations.
        peak_min_val (float): The minimum value of a peak used when calculating coordinates of object locations.

    Returns:
        peaks (list): A list of tuples containing the x, y, z coordinates of the peaks.
    """

    heatmap = tio.Image(heatmap_path, type=tio.LABEL)

    if plot:
        print('Plotting heatmap...')
        viewer = napari.view_image(heatmap.numpy(), name='heatmap')

    print('Locating peaks...')
    peaks = locate_peaks_in_volume(heatmap.numpy(), min_distance=peak_min_dist, min_val=peak_min_val)

    if save:
        print('Saving peaks...')
        peaks_path = Path(heatmap_path).with_suffix('.peaks.csv')
        np.savetxt(peaks_path, peaks, delimiter=',')

    if plot:
        print('Plotting peaks...')
        viewer.add_points(peaks, name='peaks')
        input('Press enter to continue/exit')

    return peaks
