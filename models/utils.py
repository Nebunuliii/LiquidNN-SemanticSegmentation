import os

import torch
from torch.amp import autocast, GradScaler
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm


def validate(model, device, val_dl, crit):
    model.eval()
    val_loss = 0.0

    validate_bar = tqdm(val_dl, total=len(val_dl), desc='Validating', leave=False)

    with torch.no_grad():
        for images, labels in validate_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = crit(outputs, labels)
            val_loss += loss.item()
            validate_bar.set_postfix(batch_loss=loss.item())

    avg_val_loss = val_loss / len(val_dl)
    print(f'Val Loss: {avg_val_loss:.4f}')
    return avg_val_loss


def train(model, device, train_dl, val_dl, optim, crit, save_dir, epochs=100, retries=10):
    model.to(device)
    scaler = GradScaler(device='cuda')
    counter = 0
    best_val_loss = float('inf')
    _, best_saved_val_loss = get_best_weights(save_dir)
    print('Training started...')

    for epoch in range(0, epochs):
        model.train()
        train_loss = 0.0
        print(f'Epoch: {epoch + 1}/{epochs}:')

        train_bar = tqdm(train_dl, total=len(train_dl), desc='Training', leave=False)

        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):  # enable AMP
                outputs = model(images)
                loss = crit(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            torch.cuda.synchronize()  # wait for all CUDA ops to finish

            train_loss += loss.item()
            train_bar.set_postfix(batch_loss=loss.item())

        avg_train_loss = train_loss / len(train_dl)
        print(f'Train Loss: {avg_train_loss:.4f}')

        avg_val_loss = validate(model, device, val_dl, crit)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            if best_val_loss < best_saved_val_loss:
                print(f'Saving best model at epoch {epoch + 1}...')
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f'weightval_{best_val_loss:.4f}_epoch_{epoch + 1}.pth'))
                counter = 0
        else:
            counter += 1

        if counter >= retries:
            print("Early stopping triggered.")
            break

    print('Training complete!')


def test_pixel_accuracy(model, device, test_dl):
    model.to(device)
    model.eval()
    correct_pixels = 0
    total_pixels = 0

    test_bar = tqdm(test_dl, total=len(test_dl), desc='Testing pixel accuracy', leave=False)

    with torch.no_grad():
        for images, labels in test_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct_pixels += (predictions == labels.squeeze(1)).sum().item()
            total_pixels += torch.numel(predictions)

            test_bar.set_postfix(accuracy=correct_pixels / total_pixels)

    pixel_accuracy = correct_pixels / total_pixels
    print(f'Pixel Accuracy: {pixel_accuracy:.4f}')
    return pixel_accuracy


def test_miou(model, device, test_dl, num_classes):
    model.to(device)
    model.eval()
    metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)

    test_bar = tqdm(test_dl, total=len(test_dl), desc='Testing MIoU', leave=False)

    with torch.no_grad():
        for images, labels in test_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            targets = labels.squeeze(1)

            metric.update(predictions, targets)
            current_miou = metric.compute().mean().item()
            test_bar.set_postfix(miou=current_miou)

    final_miou = metric.compute().mean().item()
    print(f'Mean IoU: {final_miou:.4f}')
    return final_miou


def get_best_weights(load_dir):
    best_filename = None
    best_val = float('inf')

    for filename in os.listdir(load_dir):
        parts = filename.split('_')

        try:
            val = float(parts[1])
        except (ValueError, IndexError):
            continue

        if val < best_val:
            best_val = val
            best_filename = filename

    return os.path.join(load_dir, best_filename).replace('\\', '/'), best_val
