import os

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # --- CONFIG ---
    input_img_dir = 'data/validation/images'
    input_lbl_dir = 'data/validation/v2.0/instances'
    output_img_dir = 'data/validation/preprocessed/images_512'
    output_lbl_dir = 'data/validation/preprocessed/labels_512'
    resize_shape = (512, 512)

    # --- CREATE OUTPUT DIRS ---
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    # --- PROCESS ---
    image_filenames = [f for f in os.listdir(input_img_dir) if f.endswith('.jpg')]

    for fname in tqdm(image_filenames, desc="Preprocessing"):
        # Paths
        base_name = os.path.splitext(fname)[0]  # removes .jpg or .JPG safely

        img_path = os.path.join(input_img_dir, fname)
        lbl_path = os.path.join(input_lbl_dir, base_name + '.png')

        out_img_path = os.path.join(output_img_dir, fname)  # keep .jpg
        out_lbl_path = os.path.join(output_lbl_dir, base_name + '.npy')

        # Skip if already processed
        if os.path.exists(out_img_path) and os.path.exists(out_lbl_path):
            continue

        # --- IMAGE ---
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue

        img_resized = cv2.resize(img, resize_shape)
        cv2.imwrite(out_img_path, img_resized)

        # --- LABEL ---
        lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if lbl is None:
            print(f"❌ Failed to read label: {lbl_path}")
            continue

        lbl = lbl // 256  # convert from instance ID to class ID
        lbl_resized = cv2.resize(lbl, resize_shape, interpolation=cv2.INTER_NEAREST)
        np.save(out_lbl_path, lbl_resized)
