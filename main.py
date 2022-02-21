import sys
import optuna
from optuna.visualization import plot_contour, plot_optimization_history
from mctnet.actions import train, inference, locate_peaks, objective


# to monitor training, run this in terminal:
# tensorboard --logdir lightning_logs

# TODO:
# see if I can quickly generate a sigma value for the gaussian noise around labels like Payer et al.
# allow the test sampler the ability to sample with a grid
# allow the outputs to be reconstructed, if using a grid sampler


if __name__ == '__main__':
    USAGE = (
        '''
        Usage:
        To train a new model and generate a checkpoint with model parameters:
        python main.py [train]
        or
        To tune a model:
        python main.py [tune] [study_name] [storage (sql storage url)]
        or
        To run inference on a volume using a specific checkpoint with model parameters:
        python main.py [inference] [volume_path] [checkpoint_path] [hparams_path] [transform_patch]
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
            print('No checkpoint argument found, loading default checkpoint')
            args.append('lightning_logs/version_31/checkpoints/epoch=181-step=706159.ckpt')

        if len(args) < 4:
            print('No hparams found, loading default haparams')
            args.append('')
        
        if len(args) < 5:
            transform_patch = True
        else:
            transform_patch = args[4]
    elif args[0] == 'tune':
        if len(args) < 2:
            study_name="crab_tuning"
            storage="sqlite:///hyperparam_tuning.db"
            print(f'No study_name argument found. Using default study_name of {study_name}')
            print(f'No storage argument found. Using default storage of {storage}')
        else:
            study_name=args[1]
            storage=args[2]

    config = {
        'lr': 0.007876941994472506,
        'weight_decay': 0,
        'momentum': 0.9264232659838044,
        'batch_size': 2,
        # TODO remove features and features_scalar
        # 'features': (64, 64, 128, 256, 512, 64),
        # 'features_scalar': 1, # multiplied by 'features' to get the feature size
        'patch_size': 64,
        'samples_per_volume': 64,
        # 'max_length': 64, 
        'max_length': 128, 
        # 'act': 'relu',
        'act': 'ReLU',
        'seed': 42,
        'train_val_ratio': 0.8,
        'train_images_dir': '/home/jake/projects/mctnet/dataset/fiddler/cropped/images/',
        'train_labels_dir': '/home/jake/projects/mctnet/dataset/fiddler/cropped/labels/',
        'test_images_dir': '/home/jake/projects/mctnet/dataset/fiddler/cropped/test_images/',
        'test_labels_dir': '/home/jake/projects/mctnet/dataset/fiddler/cropped/test_labels/',
        'num_encoding_blocks': 4,
        'out_channels_first_layer': 32,
        'pooling_type': 'avg',
        'upsampling_type': 'linear',
        'dropout': 0,
        'balanced_sampler': True,
        'debug_plots': False
    }

    if args[0] == 'train':
        train(config, show_progress=True)
    elif args[0] == 'tune':
        study = optuna.create_study(
            direction='minimize',
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10000),
            pruner=optuna.pruners.HyperbandPruner(),
            # sampler=optuna.samplers.TPESampler(),
            sampler=optuna.samplers.RandomSampler(seed=config['seed']),
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial, config, num_epochs=70), n_trials=0, gc_after_trial=True)
        print(study.best_params)
        plot_contour(study).show()
        plot_optimization_history(study).show()
    elif args[0] == 'inference':
        inference(args[3], args[2], args[1], transform_patch=transform_patch)
    elif args[0] == 'locate_peaks':
        peaks = locate_peaks(args[1])
        print(peaks)
