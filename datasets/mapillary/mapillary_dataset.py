import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MapillaryDataset(Dataset):
    def __init__(self, dataset_dir, img_size=(512, 512)):
        self.size = img_size

        images_dir = os.path.join(dataset_dir, 'preprocessed', 'images_512')
        labels_dir = os.path.join(dataset_dir, 'preprocessed', 'labels_512')

        self.images_path = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
        ])
        self.labels_path = sorted([
            os.path.join(labels_dir, f) for f in os.listdir(labels_dir)
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index])
        image = image[..., ::-1].copy()
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = np.load(self.labels_path[index])
        label = torch.from_numpy(label).long()

        return image, label

    def __len__(self):
        return len(self.images_path)
