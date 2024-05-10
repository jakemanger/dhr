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
from pathlib import Path
import numpy as np
import yaml

from optuna.integration import PyTorchLightningPruningCallback
import optuna

from deep_radiologist.inference_manager import InferenceManager
from deep_radiologist.lightning_modules import DataModule, Model
from deep_radiologist.heatmap_peaker import locate_peaks_in_volume


device = torch.device("cuda")
torch.set_float32_matmul_precision('medium')


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

    data = DataModule(config=config)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    plt.rcParams["figure.figsize"] = 12, 6
    print("TorchIO version:", tio.__version__)

    if run_internal_setup_func:
        data.setup()
        print("Training:  ", len(data.train_set))
        print("Validation: ", len(data.val_set))
        print("Test:      ", len(data.test_set))

    return data


def train(
    config,
    num_steps=400000,
    num_epochs=None,
    show_progress=False,
    starting_weights_path=None,
    profile=False,
):
    """Trains the model using hyperparameters from config

    Args:
        config (dict): The config dictionary.
        num_steps (int): The maximum number of steps to train for if early
        stopping doesn't occur.
        num_epochs (int): The maximum number of epochs to train for if early stopping
        doesn't occur.
        show_progress (bool): Whether to show a progress bar.
        starting_weights_path (str): Path to a checkpoint file to resume training
        from. See https://pytorch-lightning.readthedocs.io/en/0.8.5/weights_loading.html#restoring-training-state
        profile (bool): Whether to profile the training.
    """

    assert not (
        num_steps is not None and num_epochs is not None
    ), "Specify either num_steps or num_epochs. Not both."

    data = init_data(config)

    model = Model(config=config)

    best_models_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    every_n_epoch_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=20)

    save_path = os.path.join("logs", config["config_stem"])

    if profile:
        # profiler = pl.profilers.AdvancedProfiler(dirpath='.', filename='profile')
        profiler = pl.profilers.PyTorchProfiler(dirpath=',', filename='profile')
    else:
        profiler = None

    trainer = pl.Trainer(
        # strategy='ddp',
        accelerator='gpu',
        devices=1,
        precision=16, #'16-mixed',
        callbacks=[best_models_callback, every_n_epoch_callback],
        max_steps=num_steps,
        max_epochs=num_epochs,
        enable_progress_bar=show_progress,
        # reload_dataloaders_every_epoch=True if config["learn_sigma"] else False,
        enable_checkpointing=True,
        default_root_dir=save_path,
        profiler=profiler,
        resume_from_checkpoint=starting_weights_path
    )
    trainer.logger._default_hp_metric = False

    start = datetime.now()
    print("Training started at", start)

    trainer.fit(model=model, datamodule=data)

    print("Training duration:", datetime.now() - start)


def objective(
    trial: optuna.trial.Trial,
    config,
    num_steps,
    num_epochs=None,
    show_progress=True,
    var_to_optimise='val_1_take_f1',
    direction='minimize'
):
    """Objective function for optuna.

    Args:
        trial (optuna.trial.Trial): The trial object.
        config (dict): The config dictionary.
        num_steps (int): The number of steps to train for.
        num_epochs (int): The number of epochs to train for.
        show_progress (bool): Whether to show progress.

    Returns:
        float: The validation loss.

     Should be called with
     ```
     study.optimize(lambda trial: objective(trial, config, num_steps), n_trials=50)
     ```
    """
    assert direction in ['minimize', 'maximize']

    assert not (
        num_steps is not None and num_epochs is not None
    ), "Specify either num_steps or num_epochs. Not both."

    # set possible hyperparameters to tune
    config["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    config["weight_decay"] = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    config["dropout"] = trial.suggest_uniform("dropout", 0., 0.5)
    config["momentum"] = trial.suggest_uniform("momentum", 0., 1.)
    config["starting_sigma"] = trial.suggest_uniform("starting_sigma", 1, 5)
    config['out_channels_first_layer'] = trial.suggest_categorical('out_channels_first_layer', [16, 32, 64])
    config['optimiser'] = trial.suggest_categorical('optimiser', ['SGD', 'Adam'])
    config['upsampling_type'] = trial.suggest_categorical('upsampling_type', ['linear', 'conv'])
    config['pooling_type'] = trial.suggest_categorical('pooling_type', ['max', 'avg'])
    config['balanced_saampler'] = trial.suggest_uniform('balanced_sampler', 0.4, 1)
    config['patch_size'] = trial.suggest_categorical('patch_size', [64, 128])

    if config['learn_sigma']:
        config['sigma_regularizer'] = trial.suggest_uniform('sigma_regularizer', 1e-14, 1e-10)


    config['mse_with_f1'] = trial.suggest_categorical("mse_with_f1", [True, False])
    # config['act'] = trial.suggest_categorical('act', ['ReLU', 'LeakyReLU'])

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor=var_to_optimise,
        min_delta=0.05,
        patience=10,
        mode='min' if direction == 'minimize' else 'max'
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
    )
    pruning_callback = PyTorchLightningPruningCallback(trial, var_to_optimise)
    model = Model(config=config)
    data = init_data(config)

    save_path = os.path.join(
        "logs",
        config["config_stem"],
        "hyperparameter_tuning",
        f"version_{trial.number}",
    )

    every_n_epoch_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=20)

    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=1 if torch.cuda.is_available() else None,
        devices="auto",
        precision=16,
        callbacks=[checkpoint_callback, pruning_callback, every_n_epoch_callback],
        max_steps=num_steps,
        enable_progress_bar=show_progress,
        max_epochs=num_epochs,
        enable_checkpointing=True,
        default_root_dir=save_path,
    )

    trainer.logger.log_hyperparams(config)

    trainer.fit(model=model, datamodule=data)

    # fix for deadlock issue
    # kill all the workers when the trial has finished
    if hasattr(data, "train_queue"):
        del data.train_queue._subjects_iterable
        del data.train_queue
    if hasattr(data, "val_queue"):
        del data.val_queue._subjects_iterable
        del data.val_queue
    if hasattr(data, "test_queue"):
        del data.test_queue._subjects_iterable
        del data.test_queue
    if data in globals():
        del data

    torch.cuda.empty_cache()
    gc.collect()

    return trainer.callback_metrics[var_to_optimise].item()


def inference(
    config_path,
    checkpoint_path,
    volume_path,
    aggregate_and_save=True,
    patch_size=128,
    patch_overlap=16,
    # patch_overlap=0,
    batch_size=3,
    n_x_dirs=3,
    n_y_dirs=3,
    n_z_dirs=3,
    debug_patch_plots=False,
    debug_volume_plots=False,
):
    """Produces a plot of the model's predictions on the test set.

    Args:
        config_path (dict): Path to a hparams .yaml file with a configuration dictionary.
        checkpoint_path (str): The path to the model checkpoint.
        volume_path (str): The path to the volume to be predicted. If it is a
        directory, then all volumes in the directory will have inference ran on
        them.
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
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)["config"]
    data = init_data(config, run_internal_setup_func=True)

    model = Model.load_from_checkpoint(checkpoint_path, hparams_file=config_path).to(
        device
    )

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
            combination_mode="average",
        )

        prediction_path = "./output/" + str(
            Path(Path(volume_path).stem).with_suffix(
                "."
                + os.path.relpath(checkpoint_path)
                .replace("/", "_")
                .replace(".ckpt", "_")
                + "prediction.nii"
            )
        )

        print(f"Saving prediction to {prediction_path}")
        prediction.save(prediction_path)

        return prediction_path

    else:
        model.eval()
        with torch.no_grad():
            viewer = napari.Viewer(title="Inputs, Labels and Predictions", ndisplay=3)

            for batch in data.test_dataloader():
                # for batch in data.val_dataloader():
                inputs = batch["image"][tio.DATA].to(model.device)
                y = batch["label"][tio.DATA].to(model.device)
                pred_y = model(inputs).to(device)

                # plot
                for i in range(
                    len(inputs)
                ):  # loop though each volume in batch and plot
                    viewer.add_image(
                        inputs[i, ...].cpu().numpy(), name="x", contrast_limits=(0, 1)
                    )
                    viewer.add_image(
                        y[i, ...].cpu().numpy(), name="y", contrast_limits=(0, 1)
                    )
                    viewer.add_image(
                        pred_y[i, ...].cpu().numpy(),
                        name="y_hat",
                        contrast_limits=(0, 1),
                    )
                    y_coord = model._locate_coords(
                        y[i, ...].cpu().detach().numpy(), min_val=config['peak_min_val']
                    )
                    y_hat_coord = model._locate_coords(
                        pred_y[i, ...].cpu().detach().numpy(), min_val=config['peak_min_val']
                    )
                    viewer.add_points(
                        y_coord, name="y coordinates", size=2, face_color="green"
                    )
                    viewer.add_points(
                        y_hat_coord, name="y_hat coordinates", size=2, face_color="blue"
                    )
                    tp, fp, fn, loc_err = model._get_acc_metrics(y_hat_coord, y_coord)
                    print(f"True positives: {tp}")
                    print(f"False positives: {fp}")
                    print(f"False negatives: {fn}")
                    print(f"Localization error: {loc_err}")
                    input("Press enter to continue")
                    viewer.layers.clear()


def locate_peaks(
    heatmap_path, resample_ratio, bbox=None, save=True, plot=False, peak_min_val=0.5
):
    """Locate the peaks in a heatmap.

    Args:
        heatmap_path (str): The path to the heatmap to be processed.
        resample_ratio (float): The ratio by which to turn the predicted peaks into the original image space.
        bbox ()
        save (bool): Whether to save the results.
        plot (bool): Whether to plot the results.
        peak_min_val (float): The minimum value of a peak used when calculating coordinates of object locations.

    Returns:
        peaks (list): A list of tuples containing the x, y, z coordinates of the peaks.
    """

    heatmap = tio.Image(heatmap_path, type=tio.LABEL)

    if plot:
        print("Plotting heatmap...")
        viewer = napari.view_image(heatmap.numpy(), name="heatmap")

    print("Locating peaks...")
    peaks = locate_peaks_in_volume(
        heatmap.numpy(), min_val=peak_min_val
    )

    if plot:
        print("Plotting peaks...")
        viewer.add_points(peaks, name="peaks")
        input("Press enter to continue/exit")

    if save:
        print("Saving peaks in resampled space...")
        peaks_path = Path(heatmap_path).with_suffix(".resampled_space_peaks.csv")
        np.savetxt(peaks_path, peaks, delimiter=",")
        
        print('Converting peaks to original image space...')
        print(f'Resample ratio: {resample_ratio}')
        peaks = np.array(peaks) * resample_ratio
        print('testing new bbox')
        peaks = peaks + bbox[0] if bbox is not None else peaks
        print("Saving peaks in original image space...")
        peaks_path = Path(heatmap_path).with_suffix(".peaks.csv")
        np.savetxt(peaks_path, peaks, delimiter=",")

    return peaks
