import torch
from tqdm import tqdm
import gc
import numpy as np


def train_epoch(fold, epoch, net, optimizer, train_loader, criterion, max_ep):
    net.train(True)

    running_loss = 0.0
    n_batches = len(train_loader)
    pbar = tqdm(total=n_batches, ncols=200)
    for i, entry in enumerate(train_loader):
        inputs = entry['img'].to("cuda")
        mask = entry['mask'].to("cuda")

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f"Fold[{fold}] [{epoch} | {max_ep}] | "
                             f"Running loss {running_loss / (i + 1):.5f} / {loss.item():.5f}")
        pbar.update()

        gc.collect()
    gc.collect()
    pbar.close()

    return running_loss / n_batches


def validate_epoch(epoch, max_epoch, model, val_loader, criterion, n_classes=3):
    model.train(False)
    device = next(model.parameters()).device
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.uint32)
    val_loss = 0
    with torch.no_grad():
        for entry in tqdm(val_loader, total=len(val_loader), desc=f"[{epoch} / {max_epoch}] Val: "):
            img = entry['img'].to(device)
            mask = entry['mask'].to(device).squeeze()
            preds = model(img)
            val_loss += criterion(preds, mask).item()

            mask = mask.to('cpu').numpy()
            if n_classes == 2:
                preds = (preds.to('cpu').numpy() > 0.5).astype(float)
            elif n_classes > 2:
                preds = preds.to('cpu').numpy().argmax(axis=1)
            else:
                raise ValueError

            confusion_matrix += calculate_confusion_matrix_from_arrays(preds, mask, n_classes)

    val_loss /= len(val_loader)

    return val_loss, confusion_matrix


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    """
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_dice(confusion_matrix):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    """

    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices