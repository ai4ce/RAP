import math
import os

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# setup individual scene IDs and their download location
scenes = [
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip',
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip',
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip',
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251294/StMarysChurch.zip',
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251291/GreatCourt.zip',
]

target_height = 480  # rescale images

root = r"data/Cambridge"
os.makedirs(root, exist_ok=True)

for scene in scenes:
    scene_file = scene.split('/')[-1]
    scene_name = scene_file[:-4]
    print(f"Processing {scene_name}")

    print("Downloading and unzipping data...")
    err = os.system(f'wget {scene}')
    if err:
        print(f"Failed to download {scene}")
        continue
    err = os.system(f'unzip {scene_file}')
    if err:
        print(f"Failed to unzip {scene_file}")
        continue
    os.system(f'rm {scene_file}')
    err = os.system(f'mv {scene_name} {root}')
    if err:
        print(f"Failed to move {scene_name}")
        continue

    print("Loading SfM reconstruction...")
    datadir = f'{root}/{scene_name}'
    input_file = f'{datadir}/reconstruction.nvm'
    valid_images = set()
    with open(input_file) as f:
        f.readline()
        f.readline()
        n_images = int(f.readline().strip())
        print(f"Number of images: {n_images}")
        for i in range(n_images):
            image_path, *_ = f.readline().strip().split()
            valid_images.add(image_path)

    modes = ['train', 'test']
    for mode in modes:
        print(f"Converting {mode} data...")
        img_output_folder = f'{datadir}/{mode}/rgb'
        pose_output_folder = f'{datadir}/{mode}/poses'
        os.makedirs(img_output_folder, exist_ok=True)
        os.makedirs(pose_output_folder, exist_ok=True)
        # get list of images for current mode (train vs. test)
        image_list = f'{datadir}/dataset_{mode}.txt'
        with open(image_list) as f:
            lines = f.readlines()
        for line in tqdm(lines[3:]):
            line = line.strip()
            if not line:
                continue
            image_path, *pose = line.strip().split()
            image_name = image_path.replace(".png", ".jpg")
            if image_name not in valid_images:
                print(f"Skipping image {image_name}. Not part of reconstruction.")
                continue
            image_name = image_name.replace('/', '_')

            tx, ty, tz, qw, qx, qy, qz = map(float, pose)
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            extr = np.eye(4)
            extr[:3, :3] = R.T
            extr[:3, 3] = [tx, ty, tz]
            pose = extr
            # pose = np.linalg.inv(extr)
            if np.absolute(pose[:3, 3]).max() > 10000:
                print(f"Skipping image {image_name}. Extremely large translation. Outlier?")
                print(pose[:3, 3])
                continue
            np.savetxt(f"{pose_output_folder}/{image_name[:-3]}txt", pose)

            image = cv.imread(f"{datadir}/{image_path}")
            img_aspect = image.shape[0] / image.shape[1]
            if img_aspect > 1:
                # portrait
                img_w = target_height
                img_h = int(math.ceil(target_height * img_aspect))
            else:
                # landscape
                img_w = int(math.ceil(target_height / img_aspect))
                img_h = target_height
            image = cv.resize(image, (img_w, img_h))
            cv.imwrite(f"{img_output_folder}/{image_name}", image, [cv.IMWRITE_JPEG_QUALITY, 100])
