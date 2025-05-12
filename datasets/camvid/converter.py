import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # --- CONFIG ---
    input_img_dir = 'data/validation/val'
    input_lbl_dir = 'data/validation/val_labels'
    output_img_dir = 'data/validation/preprocessed/images_512'
    output_lbl_dir = 'data/validation/preprocessed/labels_512'
    resize_shape = (512, 512)

    # --- CREATE OUTPUT DIRS ---
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    # --- PROCESS ---
    image_filenames = [f for f in os.listdir(input_img_dir) if f.endswith('.png')]
    labels_filenames = [f for f in os.listdir(input_lbl_dir) if f.endswith('.png')]

    for fname in tqdm(image_filenames, desc='Preprocessing images'):
        img_path = os.path.join(input_img_dir, fname)
        out_img_path = os.path.join(output_img_dir, fname)

        if os.path.exists(out_img_path):
            continue

        # --- IMAGE ---
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue

        img_resized = cv2.resize(img, resize_shape)
        cv2.imwrite(out_img_path, img_resized)

    csv_path = 'data/class_dict.csv'
    with open(csv_path, mode='r') as csv_file:
        data = csv.reader(csv_file)
        next(data)
        class_dict = {
            (int(r), int(g), int(b)): int(cls) for _, r, g, b, cls in data
        }


    def attribute_classes(np_array, dict):
        h, w, _ = np_array.shape
        class_map = np.full((h, w), fill_value=255, dtype=np.uint8)

        for (r, g, b), class_id in dict.items():
            mask = (np_array[:, :, 0] == r) & (np_array[:, :, 1] == g) & (np_array[:, :, 2] == b)
            class_map[mask] = class_id

        return class_map


    for fname in tqdm(labels_filenames, desc='Preprocessing labels'):
        lbl_path = os.path.join(input_lbl_dir, fname)
        out_lbl_path = os.path.join(output_lbl_dir, fname)

        if os.path.exists(out_lbl_path):
            continue

        lbl = cv2.cvtColor(cv2.imread(lbl_path), cv2.COLOR_BGR2RGB)
        if lbl is None:
            print(f"❌ Failed to read label: {lbl_path}")
            continue
        lbl = attribute_classes(lbl, class_dict)
        lbl_resized = cv2.resize(lbl, resize_shape, interpolation=cv2.INTER_NEAREST)
        np.save(out_lbl_path, lbl_resized)
