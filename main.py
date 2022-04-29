import optuna
from optuna.visualization import plot_contour, plot_optimization_history
from deep_radiologist.actions import train, inference, locate_peaks, objective
import yaml
from yaml.loader import SafeLoader
import argparse


def main():
    # load arguments
    parser = argparse.ArgumentParser(
        description='Train, run hyperparameter tuning on, or run inference of a deep_radiologist model'
    )

    parser.add_argument(
        'mode',
        type=str,
        choices=['train', 'tune', 'infer', 'locate_peaks'],
        help='Mode of operation'
    )

    parser.add_argument(
        'config_path',
        type=str,
        help='''
        Path to your config file.
        
        Example:
            configs/fiddlercrab_corneas.yaml
        '''
    )

    parser.add_argument(
        '--volume_path',
        '-v',
        type=str,
        required=False,
        help='''
        Path to the volume to run inference on. Should have been resampled to be approximately
        the same resolution as the training data.
        In the special case when `mode` is `locate_peaks`, this should be the path to prediction
        for a previous inference run.
        '''
    )

    parser.add_argument(
        '--sql_storage_url',
        '-s',
        type=str,
        required=False,
        help='''
        Url to the sql database to store the results of hyperparameter tuning.
        Note, if you run this command using the same url in multiple terminals,
        the tuning job will become parallelised and run faster. This can also
        work across multiple machines if the sql database is accessible to both
        (e.g. with a postgresql or mysql server).

        Example:
            sqlite:///hyperparam_tuning.db
        '''
    )

    parser.add_argument(
        '--model_path',
        '-m',
        type=str,
        required=False,
        help='''
        The path to the directory containing your model.

        Example:
            zoo/fiddlercrab_corneas/version_4/
        '''
    )

    args = parser.parse_args()

    if args.mode == 'infer':
        if args.volume_path is None:
            raise Exception('Must provide a volume path to run inference on')
        if args.model_path is None:
            raise Exception('Must provide a model path to run inference with')
    elif args[0] == 'tune':
        if args.sql_storage_url is None:
            raise Exception('Must provide a sql_storage_url to store the results of tuning')
        study_name = args.config_path.split('/')[-1].split('.')[0]

    # load config
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    # start action
    if args.mode == 'train':
        train(config, show_progress=True)
    elif args.mode == 'tune':
        study = optuna.create_study(
            direction='minimize',
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10000),
            pruner=optuna.pruners.HyperbandPruner(),
            # sampler=optuna.samplers.TPESampler(),
            sampler=optuna.samplers.RandomSampler(), # dont send seed, as multiple workers will suggest identical values
            study_name=study_name,
            storage=args.sql_storage_url,
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial, config, num_epochs=70), n_trials=0, gc_after_trial=True)
        print('Best study parameters:')
        print(study.best_params)
        plot_contour(study).show()
        plot_optimization_history(study).show()
    elif args.mode == 'infer':
        hparams = f'{args.model_path}/hparams.yaml'
        checkpoint = f'{args.model_path}/checkpoints/last.ckpt'
        prediction_path = inference(config_path=hparams, checkpoint_path=checkpoint, volume_path=args.volume_path)
        peaks = locate_peaks(prediction_path, save=True, plot=True, peak_min_dist=config['peak_min_distance'], peak_min_val=config['peak_min_val'])
        print(peaks)
    elif args.mode == 'locate_peaks':
        peaks = locate_peaks(args.volume_path, save=True, plot=True, peak_min_dist=config['peak_min_distance'], peak_min_val=config['peak_min_val'])
        print(peaks)


if __name__ == '__main__':
    main()