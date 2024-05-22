"""
    Dataloader for iD datasets
    Datasets used: CIFAR10 (iD) vs SVHN/Intel Image (OoD), MNIST (iD) vs F-MNIST/K-MNIST (OoD)
"""

from tensorflow.keras.datasets import cifar10, mnist, cifar100
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import os
import numpy as np
import logging
import tensorflow as tf
from PIL import Image


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Scaling images to [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_test_org = x_test

    # Standard normalizing
    x_train = (x_train - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    x_test = (x_test - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])

    y_train = y_train[:,0]
    y_test = y_test[:,0]
    
    return x_train, y_train, x_test_org, x_test, y_test
    

def load_images_from_directory(directory, target_size=(32, 32)):
    images = []
    labels = []

    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

    for i, class_name in enumerate(classes):
        class_path = os.path.join(directory, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Read and resize image using PIL
            img = Image.open(image_path)
            img = img.resize(target_size)

            # Convert to NumPy array
            img_array = np.array(img)

            # Scaling images to [0, 1] range
            img_array = img_array.astype("float32") / 255

            # Standard normalizing
            img_array = (img_array - np.array([[[0.485, 0.456, 0.406]]])) / np.array([[[0.229, 0.224, 0.225]]])

            images.append(img_array)
            labels.append(i)

    x = np.array(images)
    y = np.array(labels)

    return x, y


def load_intel_image(data_dir='datasets/data/Intel Image'):
    tf.get_logger().setLevel(logging.ERROR)
    seg_train = os.path.join(data_dir, './seg_train/seg_train')
    seg_test = os.path.join(data_dir, './seg_test/seg_test')

    x_train, y_train = load_images_from_directory(seg_train)
    x_test, y_test = load_images_from_directory(seg_test)

    # Scaling images to [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_test_org = x_test

    # Standard normalizing
    x_train = (x_train - np.array([[[0.485, 0.456, 0.406]]])) / np.array([[[0.229, 0.224, 0.225]]])
    x_test = (x_test - np.array([[[0.485, 0.456, 0.406]]])) / np.array([[[0.229, 0.224, 0.225]]])

    return x_train, y_train, x_test_org, x_test, y_test


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Scaling images to [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_test_org = x_test

    # Retaining image shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    x_train = np.array([img_to_array(array_to_img(img).resize((32, 32))) for img in x_train])
    x_test = np.array([img_to_array(array_to_img(img).resize((32, 32))) for img in x_test])

    x_train = np.stack((x_train,) * 3, axis=-1)
    x_test = np.stack((x_test,) * 3, axis=-1)

    x_train = np.squeeze(x_train, axis=3)
    x_test = np.squeeze(x_test, axis=3)
    
    return x_train, y_train, x_test_org, x_test, y_test


def load_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    # Scaling images to [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_test_org = x_test

    # Standard normalizing
    x_train = (x_train - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    x_test = (x_test - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])

    y_train = y_train[:,0]
    y_test = y_test[:,0]
    
    return x_train, y_train, x_test_org, x_test, y_test
