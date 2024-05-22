"""
Functions for training RS-CNN
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import pickle


def train_val_split(x, y, num_classes, val_samples=-10000):
    """
    Preprocesses the input data and labels by splitting them into training and validation sets and one-hot encoding the labels.

    Parameters:
    - x: Input data.
    - y: Labels.
    - num_classes: Number of classes for one-hot encoding.
    - val_samples: Number of samples to reserve for validation. Default is -10000.

    Returns:
    - x_train: Training data.
    - y_train_one_hot: One-hot encoded labels for training data.
    - x_val: Validation data.
    - y_val_one_hot: One-hot encoded labels for validation data.
    """
    x_val = x[val_samples:]
    y_val = y[val_samples:]

    x_train = x[:val_samples]
    y_train = y[:val_samples]

    y_train_one_hot = tf.one_hot(y_train, num_classes)
    y_val_one_hot = tf.one_hot(y_val, num_classes)

    return x_train, y_train, y_train_one_hot, x_val, y_val, y_val_one_hot


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def data_generator(x):
    """
    Create and fit an ImageDataGenerator.

    Returns:
    - datagen: An ImageDataGenerator instance.
    """
    datagen = ImageDataGenerator(
        zca_epsilon=1e-06,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        horizontal_flip=True,
    )

    datagen.fit(x)

    return datagen


def lr_callbacks(lr_schedule, factor=0.1, cooldown=0, patience=5, min_lr=0.5e-6):
    """
    Create learning rate callbacks.

    Parameters:
    - lr_schedule: Learning rate schedule function.
    - factor: Factor by which the learning rate will be reduced.
    - cooldown: Number of epochs to wait before applying the learning rate reduction.
    - patience: Number of epochs with no improvement after which learning rate will be reduced.
    - min_lr: A lower bound on the learning rate.

    Returns:
    - List of learning rate callbacks.
    """
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(factor),
                                   cooldown=cooldown,
                                   patience=patience,
                                   min_lr=min_lr)

    return [lr_reducer, lr_scheduler]


def save_model_and_weights(model, selected_model, selected_dataset, model_type='CNN'):
    """
    Save the weights and model architecture of a given model.

    Parameters:
    - model: A Keras model to save.
    - selected_dataset: A string specifying the selected dataset.
    - model_type: A string specifying the model type ('CNN' or 'RSCNN').

    Returns:
    None
    """
    if model_type not in ['CNN', 'RSCNN']:
        raise ValueError("Invalid model type. Supported types are 'CNN' and 'RSCNN'.")

    model_name = f'{model_type}_{selected_model}_{selected_dataset}'
    
    # Save weights
    model_weights = model.get_weights()
    with open(f'saved_models/{model_name}_weights.pkl', 'wb') as weights_file:
        pickle.dump(model_weights, weights_file)

    # Save model
    model.save(f'saved_models/{model_name}.keras')
