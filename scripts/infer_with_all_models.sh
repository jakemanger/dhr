#!/bin/bash

# Define paired lists for configs, models, and datasets
configs_models_and_test_data=(
  # "configs/fiddlercrab_corneas.yaml ./logs/fiddlercrab_corneas/lightning_logs/version_26/ ./dataset/fiddlercrab_corneas/whole/test_images_10/"
  # "configs/fiddlercrab_rhabdoms.yaml ./logs/fiddlercrab_rhabdoms/lightning_logs/version_10/ ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/"
  # "configs/paraphronima_corneas.yaml ./logs/paraphronima_corneas/lightning_logs/version_11/ ./dataset/paraphronima_corneas/whole/test_images_10/"
  # "configs/paraphronima_rhabdoms.yaml ./logs/paraphronima_rhabdoms/lightning_logs/version_23/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
  # "configs/paraphronima_rhabdoms_without_target_heatmap_masking.yaml ./logs/paraphronima_rhabdoms_without_target_heatmap_masking/lightning_logs/version_1/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
  # # "configs/fiddlercrab_rhabdoms_with_target_heatmap_masking.yaml ./logs/fiddlercrab_rhabdoms_with_target_heatmap_masking/lightning_logs/version_2/ ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/"
  # "configs/paraphronima_rhabdoms_without_elastic_deformation.yaml ./logs/paraphronima_rhabdoms_without_elastic_deformation/lightning_logs/version_2/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
  # # "configs/paraphronima_corneas_without_elastic_deformation.yaml ./logs/paraphronima_corneas_without_elastic_deformation/lightning_logs/version_2/ ./dataset/paraphronima_corneas/whole/test_images_10/"
  # # "configs/fiddlercrab_corneas_without_elastic_deformation.yaml ./logs/fiddlercrab_corneas_without_elastic_deformation/lightning_logs/version_2/ ./dataset/fiddlercrab_corneas/whole/test_images_10/"
  # # "configs/fiddlercrab_rhabdoms_without_elastic_deformation.yaml ./logs/fiddlercrab_rhabdoms_without_elastic_deformation/lightning_logs/version_3/ ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/"
  # "configs/paraphronima_rhabdoms_without_random_scale.yaml ./logs/paraphronima_rhabdoms_without_random_scale/lightning_logs/version_3/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
  # # "configs/paraphronima_corneas_without_random_scale.yaml ./logs/paraphronima_corneas_without_random_scale/lightning_logs/version_3/ ./dataset/paraphronima_corneas/whole/test_images_10/"
  # # "configs/fiddlercrab_rhabdoms_without_random_scale.yaml ./logs/fiddlercrab_rhabdoms_without_random_scale/lightning_logs/version_3/ ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/"
  # "configs/paraphronima_rhabdoms_without_random_rotation.yaml ./logs/paraphronima_rhabdoms_without_random_rotation/lightning_logs/version_2/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
  # "configs/paraphronima_corneas_without_random_rotation.yaml ./logs/paraphronima_corneas_without_random_rotation/lightning_logs/version_1/ ./dataset/paraphronima_corneas/whole/test_images_10/"
  # "configs/fiddlercrab_corneas_without_random_rotation.yaml ./logs/fiddlercrab_corneas_without_random_rotation/lightning_logs/version_2/ ./dataset/fiddlercrab_corneas/whole/test_images_10/"
  # "configs/fiddlercrab_rhabdoms_without_random_rotation.yaml ./logs/fiddlercrab_rhabdoms_without_random_rotation/lightning_logs/version_0/ ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/"
  # "configs/fiddlercrab_corneas_hist_std.yaml ./logs/fiddlercrab_corneas_hist_std/lightning_logs/version_0/ ./dataset/fiddlercrab_corneas/whole/test_images_10/"
  # "configs/fiddlercrab_rhabdoms_hist_std.yaml ./logs/fiddlercrab_rhabdoms_hist_std/lightning_logs/version_0/ ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/"
  # "configs/fiddlercrab_corneas_random_affine_prop_1.yaml ./logs/fiddlercrab_corneas_random_affine_prop_1/lightning_logs/version_0/ ./dataset/fiddlercrab_corneas/whole/test_images_10/"
  # "configs/fiddlercrab_corneas_hist_std_z_norm.yaml ./logs/fiddlercrab_corneas_hist_std_z_norm/lightning_logs/version_0/ ./dataset/fiddlercrab_corneas/whole/test_images_10/"
  # "configs/fiddlercrab_rhabdoms_hist_std_z_norm.yaml ./logs/fiddlercrab_rhabdoms_hist_std_z_norm/lightning_logs/version_0/ ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/"
  "configs/paraphronima_corneas_from_pretrained.yaml ./logs/paraphronima_corneas_from_pretrained/lightning_logs/version_1/ ./dataset/paraphronima_corneas/whole/test_images_10/"
  "configs/paraphronima_rhabdoms_from_pretrained.yaml ./logs/paraphronima_rhabdoms_from_pretrained/lightning_logs/version_1/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
  "configs/fiddlercrab_rhabdoms_from_pretrained.yaml ./logs/fiddlercrab_rhabdoms_from_pretrained/lightning_logs/version_1/ ./dataset/fiddlercrab_rhabdoms/whole/test_images_10/"
)

nd_values=(3)
average_threshold_values=(0.5 -1 0.25)
ipmv_values=(0.5 0.25)
infer_with_max_filter_fun=(false true)

# Iterate over all combinations
for pair in "${configs_models_and_test_data[@]}"; do
  # Split the pair into config, model, and test data
  config=$(echo $pair | awk '{print $1}')
  model=$(echo $pair | awk '{print $2}')
  test_data=$(echo $pair | awk '{print $3}')
  
  for nd in "${nd_values[@]}"; do
    for max_filter_fun in "${infer_with_max_filter_fun[@]}"; do
      for average_threshold in "${average_threshold_values[@]}"; do
        for ipmv in "${ipmv_values[@]}"; do
          # Construct the command
          cmd="python main.py infer $config -v $test_data -m $model -nx $nd -ny $nd -nz $nd --average_threshold $average_threshold -ipmv $ipmv"
           
          # Optionally include --infer_with_max_filter_fun
          if [ "$max_filter_fun" == "true" ]; then
	    cmd+=" --infer_with_max_filter_fun"
          fi

          # Execute the command
          echo "Running: $cmd"
          eval $cmd
        done
      done
    done
  done
done

