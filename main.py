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
        To run inference on a volume using the default checkpoint with model parameters:
        python main.py [inference] [volume_path]
        or
        To run inference on a volume using the a specific checkpoint with model parameters:
        python main.py [inference] [volume_path] [checkpoint_path]   
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

    config = {
        'lr': 1e-2,
        'weight_decay': 0.,
        'momentum': 0.99,
        'batch_size': 2,
        'batch_size': 20,
        'features': (64, 64, 128, 256, 512, 64),
        'features_scalar': 1, # multiplied by 'features' to get the feature size
        'patch_size': 64,
        'samples_per_volume': 40,
        'max_length': 400,
        'act': 'relu',
        'seed': 42,
        'train_val_ratio': 0.8,
        'train_images_dir': '/home/jake/projects/mctnet/dataset/fiddler/images/',
        'train_labels_dir': '/home/jake/projects/mctnet/dataset/fiddler/labels/',
        'test_images_dir': '/home/jake/projects/mctnet/dataset/fiddler/test_images/',
        'test_labels_dir': '/home/jake/projects/mctnet/dataset/fiddler/test_labels/'
    }

    if args[0] == 'train':
        train(config, show_progress=True)
    elif args[0] == 'tune':
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: objective(trial, config, num_epochs=50), n_trials=100, gc_after_trial=True)
        print(study.best_params)
        plot_contour(study)
        plot_optimization_history(study)
    elif args[0] == 'inference':
        inference(config, args[1], args[2])
    elif args[0] == 'locate_peaks':
        peaks = locate_peaks(args[1])
        print(peaks)