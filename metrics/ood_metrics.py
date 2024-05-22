"""
Out-of-distribution (OoD) detection metrics: AUROC & AUPRC

Returns:
    metrics_data: AUROC, AUPRC, False Positive Rate, True Positive Rate, Precision, Recall.
"""


import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def calculate_metrics(uncertainty_iid, uncertainty_ood):
    uncertainties = np.concatenate((uncertainty_iid, uncertainty_ood))
    
    in_labels = np.zeros(uncertainty_iid.shape[0])
    ood_labels = np.ones(uncertainty_ood.shape[0])
    
    labels = np.concatenate((in_labels, ood_labels))
    
    fpr, tpr, thresholds = roc_curve(labels, uncertainties)
    auroc = roc_auc_score(labels, uncertainties)
    precision, recall, prc_thresholds = precision_recall_curve(labels, uncertainties)
    
    auprc = average_precision_score(labels, uncertainties)
    
    return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc

metrics_data = {}


def evaluate_metrics(model_name, dataset_name, labels, predictions):
    (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc = calculate_metrics(labels, predictions)
    
    if dataset_name not in metrics_data:
        metrics_data[dataset_name] = {}
    
    metrics_data[dataset_name][model_name] = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'prc_thresholds': prc_thresholds     
    }
    
    return metrics_data
    
