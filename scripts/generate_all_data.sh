#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate

# fiddlercrab eye's need to be split into subvolumes

python generate_dataset.py ./data_source_specifiers/fiddlercrab_corneas.csv -l corneas -v 10

python generate_dataset.py ./data_source_specifiers/fiddlercrab_rhabdoms.csv -l rhabdoms -v 10


# paraphronima rhabdoms don't need to be split into subvolumes (-p 0) as they are small after resampling

python generate_dataset.py ./data_source_specifiers/paraphronima_rhabdoms.csv -l rhabdoms -v 10

python generate_dataset.py ./data_source_specifiers/paraphronima_corneas.csv -l corneas -v 10 -p 0


# single label versions

python generate_dataset.py ./data_source_specifiers/fiddlercrab_corneas_single.csv -l corneas -v 10 -cl corneas

python generate_dataset.py ./data_source_specifiers/fiddlercrab_rhabdoms_single.csv -l rhabdoms -v 10 -cl rhabdoms


# paraphronima rhabdoms don't need to be split into subvolumes (-p 0) as they are small after resampling

python generate_dataset.py ./data_source_specifiers/paraphronima_rhabdoms_single.csv -l rhabdoms -v 10 -cl rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_corneas_single.csv -l corneas -v 10 -p 0 -cl corneas
