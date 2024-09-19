#!/bin/bash

# # generate all the data used in our paper
# 
source venv/bin/activate
# 
# # train from scratch models
# 
# python main.py train configs/fiddlercrab_corneas.yaml
# 
# python main.py train configs/fiddlercrab_rhabdoms.yaml
# 
# python main.py train configs/paraphronima_rhabdoms.yaml
# 
# python main.py train configs/paraphronima_corneas.yaml
# 
# 
# # train model without target heatmap masking
# 
# python main.py train configs/paraphronima_rhabdoms_without_target_heatmap_masking.yaml
# 
# # train fiddlercrab model with target heatmap masking
# 
# python main.py train configs/fiddlercrab_rhabdoms_with_target_heatmap_masking.yaml
# 
# # use base model and train from that
# 
# 
# # train models without elastic deformation
# 
# python main.py train configs/paraphronima_rhabdoms_without_elastic_deformation.yaml
# 
# python main.py train configs/paraphronima_corneas_without_elastic_deformation.yaml
# 
# python main.py train configs/fiddlercrab_corneas_without_elastic_deformation.yaml
# 
# python main.py train configs/fiddlercrab_rhabdoms_without_elastic_deformation.yaml
# 
# 
# # train models without random scale
# 
# python main.py train configs/paraphronima_rhabdoms_without_random_scale.yaml
# 
# python main.py train configs/paraphronima_corneas_without_random_scale.yaml
# 
# python main.py train configs/fiddlercrab_corneas_without_random_scale.yaml
# 
# python main.py train configs/fiddlercrab_rhabdoms_without_random_scale.yaml
# 
# 
# # train models without random rotation
# 
# python main.py train configs/paraphronima_rhabdoms_without_random_rotation.yaml
# 
# python main.py train configs/paraphronima_corneas_without_random_rotation.yaml

# python main.py train configs/fiddlercrab_corneas_without_random_rotation.yaml
# 
# python main.py train configs/fiddlercrab_rhabdoms_without_random_rotation.yaml
# 
# 
# # models with different normalisation approaches
# 
# python main.py train configs/fiddlercrab_corneas_hist_std.yaml
# 
# python main.py train configs/fiddlercrab_rhabdoms_hist_std.yaml
# 
# python main.py train configs/fiddlercrab_corneas_hist_std_z_norm.yaml
# 
# python main.py train configs/fiddlercrab_rhabdoms_hist_std_z_norm.yaml


# models starting with a pre-trained model

python main.py train configs/paraphronima_corneas_from_pretrained.yaml -w ./logs/fiddlercrab_corneas/lightning_logs/version_26/checkpoints/last.ckpt

python main.py train configs/paraphronima_rhabdoms_from_pretrained.yaml -w ./logs/fiddlercrab_corneas/lightning_logs/version_26/checkpoints/last.ckpt

python main.py train configs/fiddlercrab_rhabdoms_from_pretrained.yaml -w ./logs/fiddlercrab_corneas/lightning_logs/version_26/checkpoints/last.ckpt


