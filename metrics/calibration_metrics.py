"""
Calculate Expected Calibration Error (ECE).

Parameters:
- confidences: List or array of confidence scores predicted by the model.
- predictions: List or array of predicted class labels.
- true_labels: List or array of true class labels.
- num_bins: Number of bins for binning confidence scores.

Returns:
- ece: Expected Calibration Error.
"""

import numpy as np

def expected_calibration_error(confidences, predictions, true_labels, num_bins=10):

    # Ensure inputs are numpy arrays
    confidences = np.array(confidences)
    confidences = np.max(confidences, axis=1)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Binning confidence scores
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bins, right=True)

    # Initialize variables to store bin-wise accuracy and confidence
    bin_accuracy = np.zeros(num_bins)
    bin_confidence = np.zeros(num_bins)

    weights = np.zeros(num_bins)

    # Populate bin-wise accuracy and confidence
    for i in range(1, num_bins + 1):
        bin_mask = (bin_indices == i)
        weights[i-1] = sum(bin_mask)
        if sum(bin_mask) != 0:
            bin_accuracy[i - 1] = np.mean(np.equal(predictions[bin_mask], true_labels[bin_mask]))
            bin_confidence[i - 1] = np.mean(confidences[bin_mask])

    # Calculate ECE
    ece = np.sum(np.abs(bin_accuracy - bin_confidence) * (weights/len(true_labels)))

    return ece


def calculate_and_print_ece(predictions, y_true, model_name="Model"):
    ece_value = expected_calibration_error(predictions, np.argmax(predictions, axis=-1), y_true, 5)
    print(f"{model_name} - Expected Calibration Error (ECE): {ece_value:.4f}")