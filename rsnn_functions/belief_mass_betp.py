
"""
Belief to mass function, mass to pignistic probability

Returns:
    final_bet: Pignistic probability of predictions
"""

import numpy as np


def mass_coeff(new_classes):
    mass_co = np.zeros((len(new_classes), len(new_classes)))

    for i, A in enumerate(new_classes):
        for j, B in enumerate(new_classes):
            leng = 0
            if set(B).issubset(set(A)):
                leng = (-1) ** (len(A) - len(B))
            mass_co[j][i] = leng
    return mass_co


#Mobius inverse function
def belief_to_mass(test_preds, new_classes):
    mass_coeff_matrix = mass_coeff(new_classes)
    
    test_preds_mass = test_preds @ mass_coeff_matrix

    test_preds_mass[test_preds_mass<0] = 0
    sums_ = 1 - np.sum(test_preds_mass, axis=-1)
    sums_[sums_<0] = 0

    test_preds_mass = np.append(test_preds_mass, sums_[:, None], axis=-1)
    test_preds_mass = test_preds_mass/np.sum(test_preds_mass, axis=-1)[:, None]
    
    return test_preds_mass


# Pignistic probability
def final_betp(mass, classes, new_classes_with_full):
    betp_matrix = np.zeros((len(new_classes_with_full), len(classes)))
    for i,c in enumerate(classes): 
        for j,A in enumerate(new_classes_with_full):
            if set([c]).issubset(A):
                betp_matrix[j,i] = 1/len(A)
    
    final_bet_p = mass @ betp_matrix

    return final_bet_p