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
import napari

from mctnet.lightning_modules import DataModule, Model


# run this in terminal: tensorboard --logdir lightning_logs

seed = 42

random.seed(seed)
torch.manual_seed(seed)
num_workers = multiprocessing.cpu_count()
plt.rcParams['figure.figsize'] = 12, 6

print('TorchIO version:', tio.__version__)


data = DataModule(
    batch_size=16,
    train_val_ratio=0.8,
    train_images_dir='./dataset/crab_images/',
    train_labels_dir='./dataset/crab_labels/',
    test_images_dir='./dataset/crab_test/',
    patch_size=64,
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
    channels=(8, 16, 32, 64),
    # channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2),
)

model = Model.load_from_checkpoint(
    net=unet,
    criterion=torch.nn.MSELoss(),
    learning_rate=1e-2,
    optimizer_class=torch.optim.AdamW,
    checkpoint_path='lightning_logs/version_9/checkpoints/epoch=23-step=575.ckpt',
    hparams_file='lightning_logs/version_9/hparams.yaml',
    map_location=None
)

# test model and retrieve MSELoss values

# trainer = pl.Trainer(
#     gpus=1,
#     precision=16
# )

# start = datetime.now()
# print('Testing started at', start)
# trainer.test(model=model, datamodule=data)
# print('Testing duration:', datetime.now() - start)

# test model and retrieve Volumes

with torch.no_grad():
    # for batch in data.test_dataloader():
    for batch in data.val_dataloader():
        inputs = batch['image'][tio.DATA].to(model.device)
        y = batch['label_corneas'][tio.DATA].to(model.device)
        pred_y = model.net(inputs)

        # plot
        # with napari
        for i in range(len(inputs)): # loop though each volume in batch and plot
            viewer = napari.view_image(inputs[i, ...].cpu().numpy(), name='input')
            viewer.add_image(y[i, ...].cpu().numpy(), name='y')
            viewer.add_image(pred_y[i, ...].cpu().numpy(), name='pred_y')
            breakpoint()

        # with matplotlib
        # fig, axes = plt.subplots(3, len(inputs))
        # for i in range(len(inputs)):
        #     axes[0, i].imshow(inputs[i, 0, 128, :, :].cpu().numpy(), cmap='gray')
        #     axes[0, i].set_title('Input')
        #     axes[1, i].imshow(y[i, 0, 128, :, :].cpu().numpy(), cmap='gray')
        #     axes[1, i].set_title('Ground truth')
        #     axes[2, i].imshow(pred_y[i, 0, 128, :, :].cpu().numpy(), cmap='gray')
        #     axes[2, i].set_title('Output')
        # plt.pause(1)
        # breakpoint()


