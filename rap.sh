#!/usr/bin/env bash
config_path=configs/$1
model_path=/vast/st5265/GWOut/$2
shift 2
python rap.py --config $config_path -m $model_path $@