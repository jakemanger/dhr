#!/bin/bash

# Define paired lists for configs, models, and datasets
configs_models_and_test_data=(
  "configs/paraphronima_corneas.yaml ./logs/paraphronima_corneas/lightning_logs/version_11/ ./dataset/paraphronima_corneas/whole/test_images_10/"
  "configs/paraphronima_corneas.yaml ./logs/paraphronima_corneas/lightning_logs/version_12/ ./dataset/paraphronima_corneas/whole/test_images_10/"
  "configs/paraphronima_corneas.yaml ./logs/paraphronima_corneas/lightning_logs/version_13/ ./dataset/paraphronima_corneas/whole/test_images_10/"
   "configs/paraphronima_corneas_without_elastic_deformation.yaml ./logs/paraphronima_corneas_without_elastic_deformation/lightning_logs/version_2/ ./dataset/paraphronima_corneas/whole/test_images_10/"
   "configs/paraphronima_corneas_without_random_scale.yaml ./logs/paraphronima_corneas_without_random_scale/lightning_logs/version_3/ ./dataset/paraphronima_corneas/whole/test_images_10/"
   "configs/paraphronima_corneas_without_random_rotation.yaml ./logs/paraphronima_corneas_without_random_rotation/lightning_logs/version_1/ ./dataset/paraphronima_corneas/whole/test_images_10/"
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

