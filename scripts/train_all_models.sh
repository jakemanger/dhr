#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate

python main.py train configs/fiddlercrab_corneas.yaml

python main.py train configs/fiddlercrab_rhabdoms.yaml

python main.py train configs/paraphronima_corneas.yaml

python main.py train configs/paraphronima_rhabdoms.yaml
