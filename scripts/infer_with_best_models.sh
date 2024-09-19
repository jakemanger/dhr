#!/bin/bash

source venv/bin/activate

python main.py infer configs/fiddlercrab_corneas.yaml \
-v ./dataset/fiddlercrab_corneas/whole/test_images_10/ \
-m ./logs/fiddlercrab_corneas/lightning_logs/version_26/

python main.py infer configs/fiddlercrab_rhabdoms.yaml \
-v ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/ \
-m ./logs/fiddlercrab_rhabdoms/lightning_logs/version_10/

python main.py infer configs/paraphronima_corneas.yaml \
-v ./dataset/paraphronima_corneas/whole/test_images_10/ \
-m ./logs/paraphronima_corneas/lightning_logs/version_11/

python main.py infer configs/paraphronima_rhabdoms.yaml \
-v ./dataset/paraphronima_rhabdoms/whole/test_images_10/ \
-m ./logs/paraphronima_rhabdoms/lightning_logs/version_23/
