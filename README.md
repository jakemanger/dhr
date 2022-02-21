# MCTNet
An automatic method to detect points in micro-ct imagery by making a Convolutional neural network do heatmap regression.

## Quick start (for linux)

1. Make a python virtual environment in the root directory (if not already present).
```bash
python3.9 -m venv venv
```
See [here](https://towardsdatascience.com/getting-started-with-python-virtual-environments-252a6bd2240) 
for more information on virtual environments.

2. Activate the python virtual environment.
```bash
venv/source/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Generate the dataset (if not already found in the `./dataset/` directory)
```bash
python generate_dataset.py
```

5. Remove empty labels
```bash
python remove_empty_training_data.py
```

6. Start training (without set hyperparameters)

(If you want to watch changes in loss over time)
```bash
tensorboard --logdir lightning_logs
```

```bash
python main.py train
```

7. Start hyperparameter optimisation
Create your study name and set the url to the storage/database where you want to store it. These should be placed in the main.py file under `study_name` and `storage` variables.
Then start up processes in seperate terminals with the following command. Use as many as you can without using up your computer's resources.
This can also be done across multiple computers if then can connect to the same storage/database. I could run two processes at a time on a single machine with 128GB of RAM and 26GB VRAM.

```
python main.py tune
```
See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize for more info.

8. TODO: Run inference on a volume using a trained model
```bash
python main.py inference path/to/trained/model/directory
```

## Notes on hyperparameter tuning

Parameters for loading and sampling patches
i.e.

```
    'samples_per_volume': 32,
    'max_length': 64,
```

were manually adjusted to maximally utilise the GPU and CPU. This meant that I loaded 2 whole micro ct images per trial and then got 32 samples per volume (64 in total). This enabled me to have 3 workers, so I could run 3 trials simultaneously on my machine.

Other hyperparameters were found through hyperparameter tuning. Tuning involved using TPE sampling and median pruning in optuna to minimize Mean Squared Error. Each trial was run for 30 epochs or less if stopped early by pruning or if no improvement was observed over 5 epochs (early stopping). 


## Running my best model
### Fiddler crab
To run the current best model fiddler crab inference, type in:

#### Best manually selected hyperparameters

```
python main.py inference /home/jake/projects/mctnet/dataset/fiddler/uncropped_old/test_inference_whole/flammula_20180307.nii "/home/jake/projects/mctnet/lightning_logs_old/version_31_best_crab/checkpoints/epoch=181-step=706159 (copy).ckpt" /home/jake/projects/mctnet/lightning_logs_old/version_31_best_crab/hparams_converted_to_new.yaml
```

#### Best trial from hyperparameter training

```
python main.py inference /home/jake/projects/mctnet/dataset/fiddler/cropped/test_images_whole/dampieri_male_16-image.nii.gz /home/jake/projects/mctnet/lightning_logs/version_9/epoch=61-step=226175.ckpt /home/jake/projects/mctnet/lightning_logs/version_9/hparams.yaml
```


## File overview
```
.
├── data_info.csv  - details where annotated MATLAB files and corresponding volumes are stored
├── dataset/  - the local copy of the dataset used for analysis. This should likely be symlinked to a data drive
├── generate_dataset.py  - a script to generate your dataset
├── landmarks.npy  - a file created for histogram standardisation by torchio_data_transform.ipynb
├── lightning_logs/  - where model trial files are stored
├── main.py  - the main script to run for training, testing and inference
├── mctnet/  - helper classes and functions used specifically for this project
├── old_development_stuff/  - old files used during prototyping
├── output.png  - an example output of the model
├── README.md
├── remove_empty_training_data.py  - an experimental option to remove data with no labels
├── requirements.txt
├── scripts/  - helpful bash scripts for setup
└── torchio_data_transform.ipynb  - a file used to explore transformations of the data and generate the landmarks.npy file for histogram standardisation
```

## Known issues/possible improvements

### Training
- Labels are generated in a very slow manner using generate_dataset.py. These should probably be generated
from csv files with x, y and z coordinates when loaded during training by a custom DataLoader or similar.
Look at implementation by Payer et al.
- The GPU is often waiting for the CPU/Memory to load the next patch.
- Hyperparameters are unoptimised
- Could try 32 bit precision or mixed precision to see if it improves perfomance. Right now is at 16-bit to ensure it is fast

### Inference
- Running preprocessing transforms on the whole volume prior to inference uses a crazy amount of memory.
If there is a faster, lower memory way (e.g. running inference on each patch when loaded), then that
would be preferable.
- It's difficult to know what patch size and batch size to use.