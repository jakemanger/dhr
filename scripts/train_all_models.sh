#!/bin/bash

# generate all the data used in our paper

source venv/bin/activate

# trained from scratch models
# python main.py train configs/fiddlercrab_corneas.yaml

# python main.py train configs/fiddlercrab_rhabdoms.yaml

# python main.py train configs/paraphronima_corneas.yaml

# python main.py train configs/paraphronima_rhabdoms.yaml

# # train all models from scratch but with only that label in the dataset
# python main.py train configs/fiddlercrab_corneas_single_label_only.yaml

# python main.py train configs/fiddlercrab_rhabdoms_single_label_only.yaml

# python main.py train configs/paraphronima_corneas_single_label_only.yaml

# python main.py train configs/paraphronima_rhabdoms_single_label_only.yaml

# train all models from scratch but with only that label in the dataset with a sampler that only samples around points
python main.py train configs/fiddlercrab_corneas_single_label_only_labelled_sampler.yaml

python main.py train configs/fiddlercrab_rhabdoms_single_label_only_labelled_sampler.yaml

python main.py train configs/paraphronima_corneas_single_label_only_labelled_sampler.yaml

python main.py train configs/paraphronima_rhabdoms_single_label_only_labelled_sampler.yaml


# use base model and train from that

# python main.py train configs/fiddlercrab_rhabdoms_10.yaml -w ./logs/

# python main.py train configs/paraphronima_corneas_10.yaml

# python main.py train configs/paraphronima_rhabdoms_10.yaml
