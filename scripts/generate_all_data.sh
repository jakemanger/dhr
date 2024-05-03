#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate

<<<<<<< HEAD
python generate_dataset.py ./data_source_specifiers/fiddlercrab_corneas.csv -v 10 -p 256 -cb 32 -l corneas -cl corneas rhabdoms

python generate_dataset.py ./data_source_specifiers/fiddlercrab_rhabdoms.csv -v 10 -p 256 -cb 32 -l rhabdoms -cl corneas rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_corneas.csv -v 10 -p 256 -cb 32 -l corneas -cl corneas rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_rhabdoms.csv -v 10 -p 256 -cb 32 -l rhabdoms -cl corneas rhabdoms
=======
# fiddlercrab eye's need to be split into subvolumes

# python generate_dataset.py ./data_source_specifiers/fiddlercrab_corneas.csv -l corneas -v 10 -cl corneas

# python generate_dataset.py ./data_source_specifiers/fiddlercrab_rhabdoms.csv -l rhabdoms -v 10 -cl rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_rhabdoms.csv -l rhabdoms -v 5 10 -cl rhabdoms

python generate_dataset.py ./data_source_specifiers/paraphronima_corneas.csv -l corneas -v 10 -cl corneas
>>>>>>> last-best-merged-with-main
