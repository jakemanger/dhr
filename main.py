import sys

from torch.nn.modules import module
from mctnet.actions import train, inference, locate_peaks
from ray import tune

# to monitor training, run this in terminal:
# tensorboard --logdir lightning_logs

# TODO:
# implement hyperparameter tunings using ray tune
# add support to ensure that each image has a label for it to be sampled (and make this a hyperparameter option to learn)
# see if I can quickly generate a sigma value for the gaussian noise around labels like Payer et al.
# allow the test sampler the ability to sample with a grid
# allow the outputs to be reconstructed, if using a grid sampler

# hyperparameters taken from https://link.springer.com/chapter/10.1007/978-3-319-46723-8_27#CR12

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
        'features': (64, 64, 128, 256, 512, 64),
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
        train(config)
    elif args[0] == 'tune':
        # set possible hyperparameters to tune
        config['lr'] = tune.loguniform(1e-10, 1e-1)
        config['weight_decay'] = tune.choice([0, 1e-2, 1e-4, 1e-6])
        config['momentum'] = tune.uniform(0.9, 0.99)
        config['batch_size'] = tune.choice([1, 2, 3, 4, 5, 6])
        config['patch_size'] = tune.choice([32, 64])
        # config['features'] = tune.choice(
        #     [
        #         (32, 32, 64, 128, 256, 32),
        #         (64, 64, 128, 256, 512, 64),
        #         (128, 128, 256, 512, 1024, 128)
        #     ]
        # )

        trainable = tune.with_parameters(train)

        analysis = tune.run(
            trainable,
            resources_per_trial={'cpu': 20, 'gpu': 1},
            metric='loss',
            mode='min',
            config=config,
            num_samples=20,
            name='tune_crab_model'
        )

        print(f'Best Config: {analysis.best_config}')
        print(f'Best Trial: {analysis.best_trial}')
        print(f'Best Logdir: {analysis.best_logdir}')
        print(f'Best Checkpoint: {analysis.best_checkpoint}')
        print(f'Best Result: {analysis.best_result}')
    elif args[0] == 'inference':
        inference(config, args[1], args[2])
    elif args[0] == 'locate_peaks':
        peaks = locate_peaks(args[1])
        print(peaks)