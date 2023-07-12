#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate

python generate_dataset.py ./data_source_specifiers/fiddlercrab_corneas.csv -v 10 -p 256 -cb 32 -l corneas -cl corneas rhabdoms

python generate_dataset.py ./data_source_specifiers/fiddlercrab_rhabdoms.csv -v 10 -p 256 -cb 32 -l rhabdoms -cl corneas rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_corneas.csv -v 10 -p 256 -cb 32 -l corneas -cl corneas rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_rhabdoms.csv -v 10 -p 256 -cb 32 -l rhabdoms -cl corneas rhabdoms
