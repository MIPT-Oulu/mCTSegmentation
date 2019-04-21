import numpy as np


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
    confusion_matrix = confusion_matrix.astype(np.uint64)
    return confusion_matrix


def calculate_dice(confusion_matrix):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    """
    confusion_matrix = confusion_matrix.astype(float)
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


def calculate_iou(confusion_matrix):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    """
    confusion_matrix = confusion_matrix.astype(float)
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_volumetric_similarity(confusion_matrix):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/

    """
    confusion_matrix = confusion_matrix.astype(float)
    scores = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            vd = 0
        else:
            vd = 1 - abs(false_negatives-false_positives) / denom

        scores.append(vd)

    return scores
