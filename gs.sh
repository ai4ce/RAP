#!/usr/bin/env bash

model_root="outputs"
# model_root="/vast/st5265/GWOut"
model_name=$1
model_path="$model_root/$model_name"
data_name=$2
shift 2

python gs.py --source_path ~/7Scenes/"$data_name" --model_path "$model_path" --deblur $@
cd $model_root
~/7zz a -mmt=on -mx=9 "$model_name".7z "$model_name"
~/gdrive files upload "$model_name".7z