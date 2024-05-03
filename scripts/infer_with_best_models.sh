python main.py infer configs/fiddlercrab_rhabdoms.yaml \
-v ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/flammula_20190925_male_left-image.nii \
-m ./logs/fiddlercrab_rhabdoms/lightning_logs/version_4/ \
-b ./dataset/fiddlercrab_rhabdoms/whole/test_labels_10/flammula_20190925_male_left-bbox.csv \
-r ./dataset/fiddlercrab_rhabdoms/whole/test_labels_10/flammula_20190925_male_left-resample_ratio.txt