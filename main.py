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


def main():
    # load arguments
    parser = argparse.ArgumentParser(
        description=(
            "Train, run hyperparameter tuning on, or"
            "run inference of a deep_radiologist model"
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

    args = parser.parse_args()

    # load config
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
        config["config_stem"] = Path(args.config_path).stem

    # start action
    if args.mode == "train":
        save_path = os.path.join("logs", config["config_stem"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train(config, show_progress=True, profile=args.profile)
    elif args.mode == "tune":
        save_path = os.path.join("logs", config["config_stem"], "hyperparameter_tuning")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.sql_storage_url is None:
            args.sql_storage_url = "sqlite:///" + save_path + "/hyperparam_tuning.db"

        study_name = args.config_path.split("/")[-1].split(".")[0]
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.RandomSampler(),
            study_name=study_name,
            storage=args.sql_storage_url,
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(trial, config, num_steps=500000),
            n_trials=200,
            gc_after_trial=True,
        )
        print("Best study parameters:")
        print(study.best_params)
        plot_contour(study).show()
        plot_optimization_history(study).show()
        plot_param_importances(study).show()
    elif args.mode == "infer":
        if args.volume_path is None:
            warn(
                "Volume path (`--volume_path`) was not specified. Running inference on"
                " test patches for debugging."
            )
        if args.model_path is None:
            raise Exception("Must provide a model path to run inference with")

        if os.path.isdir(args.volume_path):
            print(
                "A directory was provided as volume_path. Looping through"
                ".nii.gz files in this directory for inference"
            )
            volumes = glob.glob(args.volume_path + "*.nii.gz")
            print(f"Found the following volumes for inference: {volumes}")
        else:
            volumes = [args.volume_path]

        for volume in volumes:
            hparams = f"{args.model_path}/hparams.yaml"
            checkpoint = f"{args.model_path}/checkpoints/last.ckpt"
            prediction_path = inference(
                config_path=hparams,
                checkpoint_path=checkpoint,
                volume_path=volume,
                aggregate_and_save=True if volume is not None else False,
            )
            peaks = locate_peaks(
                prediction_path,
                save=True,
                plot=True,
                peak_min_dist=config["peak_min_distance"],
                peak_min_val=config["peak_min_val"],
            )
            print(peaks)
    elif args.mode == "locate_peaks":
        peaks = locate_peaks(
            args.volume_path,
            save=True,
            plot=True,
            peak_min_dist=config["peak_min_distance"],
            peak_min_val=config["peak_min_val"],
        )
        print(peaks)


if __name__ == "__main__":
    main()
