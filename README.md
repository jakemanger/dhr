# MCTNet
An automatic method to detect points in micro-ct imagery 

# Quick start

1. Make a python virtual environment in the root directory (if not already present).
```bash
python3.9 -m venv venv
```

2. Activate the python virtual environment.
```bash
venv/source/activate
```

3. Generate the dataset (if not already found in the `./dataset/` directory)
```bash
python generate_dataset.py
```

3. Remove empty labels
```bash
python remove_empty_training_data.py
```

4. Start training
```bash
python main.py train
```

6. TODO: Run a trained model
```bash
python main.py inference path/to/trained/model/directory
```