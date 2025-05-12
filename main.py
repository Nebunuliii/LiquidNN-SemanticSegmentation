from torch import nn
from torch.utils.data import DataLoader

from datasets.mapillary.mapillary_dataset import MapillaryDataset
from models.resnet.resnet18 import ResNet18SemanticSegmentation
from models.utils import *

if __name__ == "__main__":
    # Ensures compatibility with Windows when using multiprocessing
    torch.multiprocessing.freeze_support()

    # Variables
    batch_size = 8
    classes = 124
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    train_dir = 'datasets\\mapillary\\data\\training'
    val_dir = 'datasets\\mapillary\\data\\validation'
    test_dir = 'datasets\\mapillary\\data\\testing'
    weights_dir = 'weights/resnet18/mapillary'

    # Datasets
    train_ds = MapillaryDataset(train_dir)
    val_ds = MapillaryDataset(val_dir)

    # Dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=10, persistent_workers=True, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True
    )

    # Model
    model = ResNet18SemanticSegmentation(classes)
    weights_path, _ = get_best_weights(weights_dir)
    model.load_state_dict(torch.load(weights_path))

    # Optim and crit
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train
    print(f'Device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')
    train(model, device, train_dl, val_dl, optimizer, criterion, save_dir=weights_dir)

    # Evaluate
    # test_pixel_accuracy(model, device, val_dl)
    # test_miou(model, device, val_dl, num_classes=classes)
