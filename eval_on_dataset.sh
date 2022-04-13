#!/bin/bash

model_path=$1
dataset_path=$2
train_config_file=${CONFIG_FILE:-config/eval.yaml}


output_result_path=$model_path/eval_on_${dataset_path##*/}
config_file=$output_result_path/eval_config.yaml

mkdir -p $output_result_path

set -e
python generate_config_eval.py \
  $model_path \
  $dataset_path \
  $train_config_file \
  $output_result_path \
  $config_file 
set +e

comm -1 -3 --nocheck-order config/template.yaml $config_file | tee $log_file

python train.py --config $config_file --test > $output_result_path/eval.log
