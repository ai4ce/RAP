#!/usr/bin/env bash

colmap feature_extractor --database_path database.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1
colmap exhaustive_matcher --database_path database.db --SiftMatching.use_gpu 1
mkdir -p sparse/orig/0
mkdir -p sparse/orig/text
mkdir -p sparse/0
colmap mapper --database_path database.db --image_path images --output_path sparse/orig
colmap model_converter --input_path sparse/orig/0 --output_path sparse/orig/text --output_type TXT
cp sparse/orig/text/cameras.txt  sparse/model
touch sparse/model/points3D.txt
colmap point_triangulator --database_path database.db --image_path images --input_path sparse/model --output_path sparse/0
colmap image_undistorter --image_path images --input_path sparse/0 --output_path undistorted
cd undistorted/sparse
mkdir 0
mv *.bin 0