#!/bin/bash

# Go to dhr/README.md


source venv/bin/activate

# infer using a paraphronima model

python main.py infer configs/cystisoma_corneas.yaml -v ./path/to/resampled/nifti -m ./zoo/paraphronima_corneas/version_8/

python main.py infer configs/cystisoma_rhabdoms.yaml -v ./path/to/resampled/nifti -m ./zoo/paraphronima_rhabdoms/version_15/


# genereate a dataset for training a new model

python generate_dataset.py ./data_source_specifiers/cystisoma_corneas.csv -l corneas -v 10 -cl corneas

python generate_dataset.py ./data_source_specifiers/cystisoma_rhabdoms.csv -l rhabdoms -v 10 -cl rhabdoms


# train a new cystisoma starting from a paraphronima model

python main.py train configs/cystisoma_corneas.yaml -w ./zoo/paraphronima_corneas/version_8/

python main.py train configs/cystisoma_rhabdoms.yaml -w ./zoo/paraphronima_rhabdoms/version_15/

python main.py infer configs/cystisoma_corneas.yaml -w ./zoo/cystisoma_corneas/version_8/


# infer using a cystisoma model

python main.py infer configs/cystisoma_corneas.yaml -v ./path/to/resampled/nifti -m ./logs/cystisoma_corneas/version_X/

python main.py infer configs/cystisoma_rhabdoms.yaml -v ./path/to/resampled/nifti -m ./zoo/cystisoma_rhabdoms/version_X/
