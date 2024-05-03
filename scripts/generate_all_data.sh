#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate

# fiddlercrab eye's need to be split into subvolumes

# python generate_dataset.py ./data_source_specifiers/fiddlercrab_corneas.csv -l corneas -v 10 -cl corneas

# python generate_dataset.py ./data_source_specifiers/fiddlercrab_rhabdoms.csv -l rhabdoms -v 10 -cl rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_rhabdoms.csv -l rhabdoms -v 5 10 -cl rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_corneas.csv -l corneas -v 10 -cl corneas
