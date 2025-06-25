#!/usr/bin/env bash

# model_root="outputs"
model_root="/vast/st5265/GWOut"
model_name=$1
model_path="$model_root/$model_name"
data_name=$2
config_path=configs/$3
shift 3

python gs.py --source_path /vast/st5265/Cambridge/"$data_name" --model_path "$model_path" --deblur $@
python rap.py --config $config_path -m $model_path