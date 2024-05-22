
import tensorflow as tf
import pickle
from rscnn_functions.rscnn_loss import BinaryCrossEntropy
import numpy as np


def get_ood_datasets(selected_dataset):
    """
    Get the Out-of-Distribution (OoD) datasets based on the selected dataset.

    Parameters:
    - selected_dataset (str): The selected dataset.

    Returns:
    - ood_datasets (list): A list of OoD datasets corresponding to the selected dataset.
    """
    # Define OoD datasets based on selected dataset
    ood_datasets_mapping = {
        "cifar10": ["ood_svhn", "ood_intel_image"],
        "mnist": ["fmnist", "kmnist"],
    }

    # Retrieve OoD datasets based on selected dataset
    ood_datasets = ood_datasets_mapping.get(selected_dataset, [])

    return ood_datasets


def load_model(selected_model, selected_dataset, model_type):
    """
    Load a Keras model from the specified model name and type.

    Parameters:
    - model_name: A string specifying the model name.
    - model_type: A string specifying the model type ('CNN' or 'RSCNN').

    Returns:
    A loaded Keras model.
    """
    if model_type not in ['CNN', 'RSCNN']:
        raise ValueError("Invalid model type. Supported types are 'CNN' and 'RSCNN'.")

    model_name = f'{model_type}_{selected_model}_{selected_dataset}'

    custom_objects = {}

    if model_type == 'RSCNN':
        custom_objects = {'BinaryCrossEntropy': BinaryCrossEntropy}
        loaded_model = tf.keras.models.load_model(f'saved_models/{model_name}.keras', custom_objects=custom_objects)
    else:
        loaded_model = tf.keras.models.load_model(f'saved_models/{model_name}.keras')

    with open(f'saved_models/{model_name}_weights.pkl', 'rb') as weights_file:
        saved_weights = pickle.load(weights_file)

    # Set weights
    loaded_model.set_weights(saved_weights)

    return loaded_model


def load_predictions(file_paths):
    return [np.load(file_path) for file_path in file_paths]

def load_all_predictions(selected_dataset):
    if selected_dataset == "mnist":
        file_paths = [
            "baselines/LB-BNN/mnist_results/mnist_iid/lbbnn_mnist_iid_preds.npy",
            "baselines/LB-BNN/mnist_results/fmnist/lbbnn_fmnist_preds.npy",
            "baselines/LB-BNN/mnist_results/kmnist/lbbnn_kmnist_preds.npy",
            "baselines/ENN/enn_preds/enn_preds/mnist_iid/enn_preds_mnist_iid.npy"
            "baselines/ENN/enn_preds/enn_preds/fashion_mnist_ood/enn_preds_fashion_mnist_ood.npy",
            "baselines/ENN/enn_preds/enn_preds/kmnist_ood/enn_preds_kmnist_ood.npy",
        ]
        
    if selected_dataset == "cifar10":
        file_paths = [
            "baselines/LB-BNN/cifar10_results/cifar10_iid/lbbnn_cifar10_iid_preds.npy",
            "baselines/LB-BNN/cifar10_results/svhn/lbbnn_svhn_preds.npy",
            "baselines/LB-BNN/cifar10_results/intel_image/lbbnn_intel_image_32_preds (1).npy",
            "baselines/ENN/enn_preds/enn_preds/cifar10_iid/enn_preds_cifar10_iid.npy",
            "baselines/ENN/enn_preds/enn_preds/svhn_ood/enn_preds_svhn_ood.npy",
            "baselines/ENN/enn_preds/enn_preds/intel_image_ood/enn_preds_intel_image_ood.npy"
        ]

    return load_predictions(file_paths)
