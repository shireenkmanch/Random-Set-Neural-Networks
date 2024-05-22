"""
Uncertainty metrics for RS-NN

Returns:
    entrop: Entropy of predictions
    max_min: Credal set width
    specificity: Pal's Specificity metric
"""



import numpy as np
import itertools
from tqdm import tqdm


# Entropy calculation
def entropy(pred_probs):
    # Avoid log(0) by setting it to a very small value
    eps = 1e-8
    entrop = -np.sum(pred_probs * np.log2(np.clip(pred_probs, eps, 1.0)), axis=1)
    return entrop


# Compute vertices of the credal set
def compute_vertices(mass, classes, class_index, new_classes_with_full):
    vertices = []
    num_permutations = 0    
    subset_permutations = []
    class_indexes = np.arange(len(classes))
    classes_array = np.array(classes)
    for _ in range(num_permutations):
      np.random.shuffle(class_indexes)
      subset_permutations.append(list(classes_array[class_indexes]))

    p_0 = classes.copy()
    p_0[0], p_0[class_index] = p_0[class_index], p_0[0]
    
    p_1 = classes.copy()
    p_1[-1], p_1[class_index] = p_1[class_index], p_1[-1]
    
    subset_permutations = [p_0] + [p_1] + subset_permutations
    
    for perm in subset_permutations:
        p = 0
        i, curr_elem = perm.index(classes[class_index]), classes[class_index]
        for c in new_classes_with_full:
          if curr_elem in c:
            for k in range(i):
              if perm[k] in c:
                break
            else:
              p += mass[new_classes_with_full.index(c)]
        vertices.append(p)
           
    return np.array(vertices)


# Credal set width
def credal_set_width(vertices):
    max_min = {}
    for c in range(len(vertices)):
        temp = np.max(vertices[c], axis=-1) - np.min(vertices[c], axis=-1)
        max_min[c] = temp
    
    return max_min


# Specificity by Pal
def specificity_metric(test_preds_mass, new_classes_with_full):
    specificity = []
    mass_cardinality_greater = []
    indices_top_card = []
    for i in range(len(test_preds_mass)):
        spec = 0
        for j,A in enumerate(new_classes_with_full):
                spec += test_preds_mass[i][j] / len(A)
        specificity.append(spec)

    for i in range(len(test_preds_mass)):
        top_labels_mass = np.argsort(test_preds_mass[i])[::-1][:70]
        #t = top_labels_mass[0] 
        check = False
        for t in top_labels_mass:
            if len(new_classes_with_full[t]) == 0:
                check = True
                break
        if not check:
            mass_card = test_preds_mass[i]
            mass_cardinality_greater.append(mass_card)
            indices_top_card.append((i, top_labels_mass[0], len(new_classes_with_full[top_labels_mass[0]]), new_classes_with_full[top_labels_mass[0]]))

    mass_cardinality_greater = np.array(mass_cardinality_greater)
    indices_top_card = np.array(indices_top_card)

    return specificity, indices_top_card
