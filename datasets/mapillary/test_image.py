import json
import os
import random

import cv2
import numpy as np
import torch
from torchvision import transforms

from models.resnet.resnet18 import ResNet18SemanticSegmentation
from models.utils import get_best_weights


def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    # Assign a color to each pixel based on its class label
    for label_id, label in enumerate(labels):
        color_array[image_array == label_id] = label["color"]

    return color_array


if __name__ == '__main__':
    # variables
    classes = 124
    alfa = 0.2
    transform = transforms.ToTensor()

    # Paths
    weights_dir = '../../weights/resnet18/mapillary'
    images_dir = 'data/testing/images'

    # Load label definitions
    with open('data/config_v2.0.json') as config_file:
        config = json.load(config_file)
    lbl = config['labels']

    # Load test images
    images_names = os.listdir(images_dir)
    images_path = [os.path.join(images_dir, f) for f in images_names]

    # Load model and weights
    model = ResNet18SemanticSegmentation(num_classes=classes)
    weights_path, _ = get_best_weights(weights_dir)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    while True:
        # Choose a random image
        random_image_path = random.choice(images_path)
        image = cv2.imread(random_image_path)

        # Convert for model
        image_resized = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image)

        # Inference
        with torch.no_grad():
            result = model(image_tensor.unsqueeze(0))
            result = torch.argmax(result, dim=1).squeeze(0)
            result = result.cpu().numpy().astype(np.uint8)

        # Convert to colored mask
        output_image = apply_color_map(result, lbl)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        # Blend prediction with input
        blended = cv2.addWeighted(image_resized, alfa, output_image, 1 - alfa, 0)

        # Show result
        cv2.imshow('Prediction', blended)
        key = cv2.waitKey(0)

        if key == 32:  # SPACE
            cv2.destroyWindow('Prediction')
            continue
        else:
            break

    cv2.destroyAllWindows()
