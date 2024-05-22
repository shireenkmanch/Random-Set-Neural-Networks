"""
Compute correct and incorrect indices given model predictions and true labels.
Compute confidence scores for given indices from model predictions.

Parameters:
- predictions: Array of model predictions.
- true_labels: Array of true labels.
- indices: Indices for which confidence scores need to be computed.

Returns:
- correct_idx: Indices of correctly classified samples.
- incorrect_idx: Indices of incorrectly classified samples.
- confidence_scores: Confidence scores for the specified indices.
"""


import numpy as np


def compute_correct_incorrect_indices(predictions, true_labels):
    predicted_labels = np.argmax(predictions, axis=-1)
    correct_idx = np.nonzero(predicted_labels == true_labels)[0]
    incorrect_idx = np.nonzero(predicted_labels != true_labels)[0]
    return correct_idx, incorrect_idx


def compute_confidence_scores(predictions, indices):
    return np.max(predictions[indices], axis=1)
