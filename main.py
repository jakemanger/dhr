import sys
import optuna
from optuna.visualization import plot_contour, plot_optimization_history
from mctnet.actions import train, inference, locate_peaks, objective
import yaml
from yaml.loader import SafeLoader


# to monitor training, run this in terminal:
# tensorboard --logdir lightning_logs


if __name__ == '__main__':
    # load arguments
    USAGE = (
        '''
        Usage:
        To train a new model and save it:
        python main.py [train]
        or
        To tune a model and find the best hyperparameters:
        python main.py [tune] [study_name] [Optional(sql_storage_url))]
        or
        To run inference on a volume using a saved model:
        python main.py [inference] [volume_path] [Optional(model_dir)] [Optional(transform_each_patch)]
        '''
    )
    args = sys.argv[1:]

    if not args or args[0] not in ['train', 'tune', 'inference', 'locate_peaks']:
        raise SystemExit(USAGE)

    if args[0] == 'inference':
        if len(args) < 2:
            print('No volume_path argument found')
            raise SystemExit(USAGE)

        if len(args) < 3:
            print('No model directory argument found, loading default model directory')
            args.append('./lightning_logs/version_8')
        
        if len(args) < 4:
            transform_patch = True
        else:
            transform_patch = args[3] in ('True', 'TRUE', 'true', '1', 't', 'T')
    elif args[0] == 'tune':
        if len(args) < 2:
            study_name="crab_tuning"
            storage="sqlite:///hyperparam_tuning.db"
            print(f'No study_name argument found. Using default study_name of {study_name}')
            print(f'No sql_storage_url argument found. Using default storage of {storage}')
        else:
            study_name=args[1]
            storage=args[2]


    # load config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    

    # start action
    if args[0] == 'train':
        train(config, show_progress=True)
    elif args[0] == 'tune':
        study = optuna.create_study(
            direction='minimize',
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10000),
            pruner=optuna.pruners.HyperbandPruner(),
            # sampler=optuna.samplers.TPESampler(),
            sampler=optuna.samplers.RandomSampler(), # dont send seed, as multiple workers will suggest identical values
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial, config, num_epochs=70), n_trials=50, gc_after_trial=True)
        print('Best study parameters:')
        print(study.best_params)
        plot_contour(study).show()
        plot_optimization_history(study).show()
    elif args[0] == 'inference':
        hparams = f'{args[2]}/hparams.yaml'
        checkpoint = f'{args[2]}/checkpoints/last.ckpt'
        prediction_path = inference(config_path=hparams, checkpoint_path=checkpoint, volume_path=args[1], transform_patch=transform_patch)
        peaks = locate_peaks(prediction_path, save=True, plot=True, peak_min_dist=config['peak_min_distance'], peak_min_val=config['peak_min_val'])
        print(peaks)
    elif args[0] == 'locate_peaks':
        peaks = locate_peaks(args[1], save=True, plot=True, peak_min_dist=config['peak_min_distance'], peak_min_val=config['peak_min_val'])
        print(peaks)
