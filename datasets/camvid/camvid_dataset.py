import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CamVidDataset(Dataset):
    def __init__(self, dataset_dir, img_size=(512, 512)):
        self.size = img_size

        images_dir = os.path.join(dataset_dir, 'preprocessed', 'images_512')
        labels_dir = os.path.join(dataset_dir, 'preprocessed', 'labels_512')

        images_path = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
        ])
        labels_path = sorted([
            os.path.join(labels_dir, f) for f in os.listdir(labels_dir)
        ])

        self.images = []
        self.labels = []

        for img_path, lbl_path in zip(images_path, labels_path):
            image = cv2.imread(img_path)
            image = image[..., ::-1].copy()
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            label = np.load(lbl_path)
            label = torch.from_numpy(label).long()

            self.images.append(image)
            self.labels.append(label)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
