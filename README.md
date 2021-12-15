# MCTNet
An automatic method to detect points in micro-ct imagery by making Convolutional neural networks do heatmap regression.

# Quick start (for linux)

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

6. Start training

(If you want to watch changes in loss over time)
```bash
tensorboard --logdir lightning_logs
```

```bash
python main.py train
```

7. TODO: Run inference on a volume using a trained model
```bash
python main.py inference path/to/trained/model/directory
```

# Known issues

## Training
- The GPU is often waiting for the CPU/Memory to load the next patch.
- Hyperparameters are unoptimised

## Inference
- Running preprocessing transforms on the whole volume prior to inference uses a crazy amount of memory.
If there is a faster, lower memory way (e.g. running inference on each patch when loaded), then that
would be preferable.
- It's difficult to know what patch size and batch size to use.