#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate

python generate_dataset.py ./data_source_specifiers/fiddlercrab_corneas.csv -l corneas

python generate_dataset.py ./data_source_specifiers/fiddlercrab_rhabdoms.csv -l rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_corneas.csv -l corneas

python generate_dataset.py ./data_source_specifiers/paraphronima_rhabdoms.csv -l rhabdoms
