"""
    Pre-trained Models for Transfer Learning: ResNet50, VGG16
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input


def pretrained_resnet50(input_shape, num_classes, final_activation):
    inputs = tf.keras.layers.Input(shape=input_shape)
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')(resize)

    base_model.trainable = False

    x = base_model(inputs, training = False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=final_activation, name="classification")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def pretrained_vgg16(input_shape, num_classes, final_activation):
    inputs = tf.keras.layers.Input(shape=input_shape)
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')(resize)

    base_model.trainable = False

    x = base_model(inputs, training = False)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=final_activation, name="classification")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

