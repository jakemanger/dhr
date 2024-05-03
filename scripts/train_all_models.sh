#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate

<<<<<<< HEAD
=======
# trained from scratch models

python main.py train configs/paraphronima_rhabdoms.yaml

python main.py train configs/paraphronima_corneas.yaml

>>>>>>> last-best-merged-with-main
python main.py train configs/fiddlercrab_corneas.yaml

python main.py train configs/fiddlercrab_rhabdoms.yaml

<<<<<<< HEAD
python main.py train configs/paraphronima_corneas.yaml

python main.py train configs/paraphronima_rhabdoms.yaml
=======
# use base model and train from that

python main.py train configs/fiddlercrab_rhabdoms.yaml -w ./logs/fiddlercrab_corneas/lightning_logs/version_4/checkpoints/last.ckpt

python main.py train configs/paraphronima_corneas.yaml -w ./logs/fiddlercrab_corneas/lightning_logs/version_4/checkpoints/last.ckpt

python main.py train configs/paraphronima_rhabdoms.yaml -w ./logs/fiddlercrab_corneas/lightning_logs/version_4/checkpoints/last.ckpt

python main.py train configs/paraphronima_rhabdoms_5.yaml -w ./logs/fiddlercrab_corneas/lightning_logs/version_4/checkpoints/last.ckpt

python main.py train configs/paraphronima_rhabdoms_15.yaml -w ./logs/fiddlercrab_corneas/lightning_logs/version_4/checkpoints/last.ckpt
>>>>>>> last-best-merged-with-main
