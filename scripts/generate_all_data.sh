#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate
python generate_dataset.py ./data_source_specifiers/fiddlercrab_corneas.csv -v 10 15 20 -l corneas
python generate_dataset.py ./data_source_specifiers/fiddlercrab_rhabdoms.csv -v 5 10 15 20 -l rhabdoms
python generate_dataset.py ./data_source_specifiers/paraphronima_corneas.csv -v 20 25 30 -l corneas
python generate_dataset.py ./data_source_specifiers/paraphronima_rhabdoms.csv -v 5 -l rhabdoms