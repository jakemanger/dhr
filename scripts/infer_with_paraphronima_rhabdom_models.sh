#!/bin/bash

# Define paired lists for configs, models, and datasets
configs_models_and_test_data=(
#   "configs/paraphronima_rhabdoms.yaml ./logs/paraphronima_rhabdoms/lightning_logs/version_24/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
#   "configs/paraphronima_rhabdoms.yaml ./logs/paraphronima_rhabdoms/lightning_logs/version_25/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
#   "configs/paraphronima_rhabdoms.yaml ./logs/paraphronima_rhabdoms/lightning_logs/version_26/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
#   "configs/paraphronima_rhabdoms.yaml ./logs/paraphronima_rhabdoms_thresholded_heatmaps/lightning_logs/version_15/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
  "configs/paraphronima_rhabdoms_whole_scans.yaml ./logs/paraphronima_rhabdoms_whole_scans/lightning_logs/version_1/ ./dataset/paraphronima_rhabdoms/whole/test_images_10/"
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

