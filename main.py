import optuna
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
)
from deep_radiologist.actions import train, inference, locate_peaks, objective
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import argparse
import os
import glob
from warnings import warn
import numpy as np


def main():
    # load arguments
    parser = argparse.ArgumentParser(
        description=(
            "Train, hyperparameter tune or "
            "infer with a deep heatmap regression model"
        )
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "tune", "infer", "locate_peaks"],
        help="Mode of operation",
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="""
        Path to your config file.
        Example:
            configs/fiddlercrab_corneas.yaml
        """,
    )

    parser.add_argument(
        "--volume_path",
        "-v",
        type=str,
        required=False,
        help="""
        Path to the volume to run inference on. Should have been resampled to
        be approximately the same resolution as the training data.
        In the special case when `mode` is `locate_peaks`, this should be the
        path to prediction for a previous inference run.
        """,
    )

    parser.add_argument(
        "--sql_storage_url",
        "-s",
        type=str,
        required=False,
        help="""
        Optionally specify the url to the sql database to store the results of hyperparameter tuning.
        Note, if you run this command using the same url in multiple terminals,
        the tuning job will become parallelised and run faster. This can also
        work across multiple machines if the sql database is accessible to both
        (e.g. with a postgresql or mysql server).
        If not specified, this will default to a sqlite database in the
        `logs/YOUR_CONFIG_FILENAME/hyperparameter_tuning/hyperparameter_tuning.db` directory.

        Example:
            sqlite:///logs/my_custom_hyperparam_tuning.db
        """,
    )

    parser.add_argument(
        "--starting_weights_path",
        "-w",
        type=str,
        required=False,
        help="""
        Path to the weights to begin training with.
        If not specified, the weights will start out randomised.

        Example:
            logs/fiddlercrab_corneas/lightning_logs/version_1/checkpoints/last.ckpt
        """,
    )

    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=False,
        help="""
        The path to the directory containing your model.

        Example:
            zoo/fiddlercrab_corneas/version_4/
        """,
    )

    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="""
        If specified, the model will be trained with a profiler to find performance bottlenecks in code.
        This only works with the `train` mode.
        See https://pytorch-lightning.readthedocs.io/en/1.4.4/advanced/profiler.html
        for more details.
        """,
    )

    default_num_steps = 400000
    parser.add_argument(
        "--num_steps",
        "-n",
        type=int,
        required=False,
        default=default_num_steps,
        help=(
            "The number of steps to train for. If specified in the config file and this argument, an error will be raised."
            f"If not specified in the config file or here, a default of {default_num_steps} steps will be used."
        )
    )

    parser.add_argument(
        '--resample_ratio_path',
        '-r',
        type=str,
        required=False,
        help="""
        Path to the resample ratio file. This file should be a .txt file with the ratio to convert predicted coordinates to the original volume space.
        The file should contain a single float value.
        """,
    )

    parser.add_argument(
        '--bbox_path',
        '-b',
        type=str,
        required=False,
        help="""
        Path to the bbox file. This file should be a .csv file with the bbox to convert predicted coordinates to the original volume space.
        The file should contain a six float values.
        """,
    )

    args = parser.parse_args()

    # load config
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
        config["config_stem"] = Path(args.config_path).stem

    # handle special argument conditions
    if 'num_steps' in config and args.num_steps != default_num_steps:
        raise AttributeError(
            '`num_steps` was specified in both the config file and a command line '
            'argument. Specify this either in a config file or as a command line '
            'argument but not both.'
        )
    num_steps = args.num_steps if 'num_steps' not in config else config['num_steps']

    # start action
    if args.mode == "train":
        save_path = os.path.join("logs", config["config_stem"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train(
            config,
            show_progress=True,
            profile=args.profile,
            num_steps=num_steps,
            starting_weights_path=args.starting_weights_path
        )
    elif args.mode == "tune":
        save_path = os.path.join("logs", config["config_stem"], "hyperparameter_tuning")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.sql_storage_url is None:
            args.sql_storage_url = "sqlite:///" + save_path + "/hyperparam_tuning.db"

        study_name = args.config_path.split("/")[-1].split(".")[0]

        direction = 'minimize'

        study = optuna.create_study(
            direction=direction,
            pruner=optuna.pruners.HyperbandPruner(max_resource=30), # from testing with manual parameters, less than 30 epochs can give great results
            sampler=optuna.samplers.TPESampler(),
            study_name=study_name,
            storage=args.sql_storage_url,
            load_if_exists=True,
        )
        n_trials = 0
        print(
            f'Optimising hyperparameters by training {n_trials} trials of different '
            f'hyperparameters for {num_steps} steps'
        )

        study.optimize(
            lambda trial: objective(trial, config, num_steps=num_steps, direction=direction),
            n_trials=n_trials,
            gc_after_trial=True,
        )
        print("Best study parameters:")
        print(study.best_params)

        plot_contour(study).show()
        plot_optimization_history(study).show()
        plot_param_importances(study).show()
        breakpoint()
    elif args.mode == "infer":
        if args.volume_path is None:
            warn(
                "Volume path (`--volume_path`) was not specified. Running inference on"
                " test patches for debugging."
            )
            volumes = [None]
        elif os.path.isdir(args.volume_path):
            print(
                "A directory was provided as volume_path. Looping through"
                ".nii files in this directory for inference"
            )
            volumes = glob.glob(args.volume_path + "*.nii")
            print(f"Found the following volumes for inference: {volumes}")
        else:
            volumes = [args.volume_path]

        if args.model_path is None:
            raise Exception("Must provide a model path to run inference with")

        for volume in volumes:
            hparams = os.path.join(args.model_path, 'hparams.yaml')
            checkpoint = os.path.join(args.model_path, 'checkpoints', 'last.ckpt')
            if not os.path.isfile(checkpoint):
                # if not labelled as last, find last editted file
                checkpoint = os.path.join(args.model_path, 'checkpoints', '*ckpt')
                list_of_files = glob.glob(checkpoint)
                checkpoint = max(list_of_files, key=os.path.getctime)

            prediction_path = inference(
                config_path=hparams,
                checkpoint_path=checkpoint,
                volume_path=volume,
                aggregate_and_save=True if volume is not None else False,
            )
            # read the resample ratio from the txt file
            resample_ratio = 1.0
            with open(args.resample_ratio_path, "r") as f:
                resample_ratio = float(f.read())

            # read the bbox from the csv file using numpy
            bbox = None
            with open(args.bbox_path, "r") as f:
                bbox = np.loadtxt(f, delimiter=",")

            peaks = locate_peaks(
                prediction_path,
                resample_ratio,
                bbox,
                save=True,
                plot=True,
                peak_min_val=config["peak_min_val"],
            )
            print(peaks)
    elif args.mode == "locate_peaks":
        peaks = locate_peaks(
            args.volume_path,
            resample_ratio=1,
            save=True,
            plot=True,
            peak_min_val=config["peak_min_val"],
        )
        print(peaks)


if __name__ == "__main__":
    main()
