# Define paths
$colmapPath = "colmap"
$databasePath = "database.db"
$imagePath = "images"
$outputSparseOrig = "sparse\orig"
$outputSparseOrigText = "$outputSparseOrig\text"
$outputSparseModel = "sparse\model"
$outputSparse0 = "sparse\0"
$undistortedSparse0 = "undistorted\sparse\0"

# Run feature extraction
& $colmapPath feature_extractor --database_path $databasePath --image_path $imagePath --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1

# Run exhaustive matching
& $colmapPath exhaustive_matcher --database_path $databasePath --SiftMatching.use_gpu 1

# Create necessary directories
New-Item -Path "$outputSparseOrig\0" -ItemType Directory -Force
New-Item -Path "$outputSparseOrigText" -ItemType Directory -Force
New-Item -Path $outputSparse0 -ItemType Directory -Force

# Create empty points3D.txt file
New-Item -Path "$outputSparseModel\points3D.txt" -ItemType File -Force

# Run mapper
& $colmapPath mapper --database_path $databasePath --image_path $imagePath --output_path $outputSparseOrig

# Convert model to text format
& $colmapPath model_converter --input_path "$outputSparseOrig\0" --output_path $outputSparseOrigText --output_type TXT

# Copy cameras.txt to model directory
Copy-Item -Path "$outputSparseOrigText\cameras.txt" -Destination $outputSparseModel

# Run point triangulation
& $colmapPath point_triangulator --database_path $databasePath --image_path $imagePath --input_path $outputSparseModel --output_path $outputSparse0

# Run distortion
& $colmapPath image_undistorter --image_path $imagePath --input_path $outputSparse0 --output_path undistorted

# Move to 0
New-Item -Path $undistortedSparse0 -ItemType Directory -Force
Get-ChildItem -Path "undistorted\sparse" -File | Move-Item -Destination $undistortedSparse0
