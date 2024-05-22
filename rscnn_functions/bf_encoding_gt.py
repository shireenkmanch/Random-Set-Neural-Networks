""" 
Belief encoding of budgeted focal sets

Returns:
    y_encoded: 2D-array of encoded ground truth
"""

import numpy as np

# Modifying the ground truth with belief encoding
def groundtruthmod(y, classes, new_classes, dict):
    y_encoded = np.zeros((len(y), len(new_classes)), dtype=int)
    for i, label in enumerate(y):
        for j, class_ in enumerate(new_classes):
            if class_.issubset(set(classes)) and dict[label] in class_:
                y_encoded[i, j] = 1
    return y_encoded