"""
    Dataloader for OoD datasets
    Datasets used: CIFAR10 (iD) vs SVHN/Intel Image (OoD), MNIST (iD) vs F-MNIST/K-MNIST (OoD)
"""

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from data.id_dataloader import load_images_from_directory
import scipy.io
import os
import numpy as np
import logging
import tensorflow as tf
import warnings
import zipfile

def load_ood_data(selected_dataset, dataset_loader):
    x_test_ood_datasets = {}
    y_test_ood_datasets = {}
    ood_datasets = {}

    if selected_dataset == "cifar10":
        ood_dataset_1 = "ood_svhn"
        ood_dataset_2 = "ood_intel_image"
    elif selected_dataset == "mnist":
        ood_dataset_1 = "ood_fashion_mnist"
        ood_dataset_2 = "ood_kmnist"
    else:
        raise ValueError(f"Invalid selected dataset: {selected_dataset}")

    ood_dataset_names = [ood_dataset_1, ood_dataset_2]

    for ood_dataset_name in ood_dataset_names:
        loader_function = dataset_loader.get(ood_dataset_name)
        if loader_function is not None:
            x_test_ood, y_test_ood = loader_function()
            x_test_ood_datasets[f'x_test_{ood_dataset_name}'] = x_test_ood
            y_test_ood_datasets[f'y_test_{ood_dataset_name}'] = y_test_ood
            ood_datasets[ood_dataset_name] = {'x_test': x_test_ood, 'y_test': y_test_ood}

    return ood_datasets


def load_ood_svhn():
    test_data = scipy.io.loadmat('data/datasets/SVHN/test_32x32.mat')

    x_test_svhn = test_data['X']
    y_test_svhn = test_data['y']
    x_test_svhn = x_test_svhn.transpose(3, 0, 1, 2)
    
    # Scaling images to [0, 1] range
    x_test_svhn = x_test_svhn.astype("float32") / 255

    # Standard normalizing
    x_test_svhn = (x_test_svhn - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    y_test_svhn = y_test_svhn[:,0]
    x_test_svhn = x_test_svhn[0:10000]
    y_test_svhn = y_test_svhn[0:10000]
    
    return x_test_svhn, y_test_svhn

    
def load_ood_intel_image():
    seg_test = 'data/datasets/Intel Image/seg_test'

    x_test_intel_image, y_test_intel_image = load_images_from_directory(seg_test)
    x_test_intel_image, y_test_intel_image = load_images_from_directory(seg_test)

    # Scaling images to [0, 1] range
    x_test_intel_image = x_test_intel_image.astype("float32") / 255

    # Standard normalizing
    x_test_intel_image = (x_test_intel_image - np.array([[[0.485, 0.456, 0.406]]])) / np.array([[[0.229, 0.224, 0.225]]])
    
    return x_test_intel_image, y_test_intel_image


def load_ood_fashion_mnist():
    (x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()
    
    # Scaling images to [0, 1] range
    x_test_fmnist = x_test_fmnist.astype("float32") / 255

    # Retaining image shape (28, 28, 1)
    x_test_fmnist = np.expand_dims(x_test_fmnist, -1)
    
    x_test_fmnist = np.array([img_to_array(array_to_img(img).resize((32, 32))) for img in x_test_fmnist])
    x_test_fmnist = np.stack((x_test_fmnist,) * 3, axis=-1)
    x_test_fmnist = np.squeeze(x_test_fmnist, axis=3)
    
    return x_test_fmnist, y_test_fmnist


def load_ood_kmnist():
    x_test_kmnist = np.load("datasets/data/K-MNIST/kmnist-test-imgs.npz")['arr_0']
    y_test_kmnist = np.load("datasets/data/K-MNIST/kmnist-test-labels.npz")['arr_0']

    # Scaling images to [0, 1] range
    x_train_kmnist = x_train_kmnist.astype("float32") / 255
    x_test_kmnist = x_test_kmnist.astype("float32") / 255

    # Retaining image shape (28, 28, 1)
    x_train_kmnist = np.expand_dims(x_train_kmnist, -1)
    x_test_kmnist = np.expand_dims(x_test_kmnist, -1)
    
    x_test_kmnist = np.array([img_to_array(array_to_img(img).resize((32, 32))) for img in x_test_kmnist])
    x_test_kmnist = np.stack((x_test_kmnist,) * 3, axis=-1)
    x_test_kmnist = np.squeeze(x_test_kmnist, axis=3)
    
    return x_test_kmnist, y_test_kmnist
