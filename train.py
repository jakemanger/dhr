import random
import multiprocessing
from datetime import datetime
import torch
import torchio as tio
import pytorch_lightning as pl
import torch.nn
import monai
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from mctnet.lightning_modules import DataModule, Model

# to monitor training, run this in terminal:
# tensorboard --logdir lightning_logs

# TODO:
# implement patch_size sampling with the sampler
# allow the test sampler the ability to sample with a grid
# allow the outputs to be reconstructed, if using a grid sampler

train = True
# else test

# hyperparameters taken from https://link.springer.com/chapter/10.1007/978-3-319-46723-8_27#CR12
config = {
    'lr': 1e-16,
    'weight_decay': 0.0005,
    'momentum': 0.99,
    'batch_size': 2,
    'patch_size': 64
}

seed = 42

random.seed(seed)
torch.manual_seed(seed)
num_workers = multiprocessing.cpu_count()
plt.rcParams['figure.figsize'] = 12, 6

print('TorchIO version:', tio.__version__)


data = DataModule(
    batch_size=config['batch_size'],
    train_val_ratio=0.8,
    train_images_dir='./dataset/crab_images/',
    train_labels_dir='./dataset/crab_labels/',
    test_images_dir='./dataset/crab_test_images/',
    test_labels_dir='./dataset/crab_test_labels/',
    patch_size=config['patch_size'],
    samples_per_volume=10,
    max_length=300,
    num_workers=10
)

data.prepare_data()
data.setup()
print('Training:  ', len(data.train_set))
print('Validation: ', len(data.val_set))
print('Test:      ', len(data.test_set))

# TODO: check what 16bit precision is
unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
)

model = Model(
    net=unet,
    criterion=torch.nn.MSELoss(),
    optimizer_class=torch.optim.SGD,
    config=config
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)
trainer = pl.Trainer(
    gpus=1,
    precision=16,
    callbacks=[early_stopping],
)
trainer.logger._default_hp_metric = False

if train:
    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)
