"""
    Model Architectures for RS-CNN: ResNet50, WideResNet-28-10, VGG16, Inception V3, EfficientNetB2
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, EfficientNetB2
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input


def residual_block(x, filters, stride, mod):
    shortcut = x

    # First convolution layer
    x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    if mod:
        x = ReLU()(x)
    else:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Second convolution layer
    x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])

    return x


def resnet50(input_shape, num_classes, final_activation):
    inputs = tf.keras.layers.Input(shape=input_shape)
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights=None)(resize)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=final_activation, name="classification")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def vgg16(input_shape, num_classes, final_activation):
    inputs = tf.keras.layers.Input(shape=input_shape)
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=(224, 224, 3), include_top=False, weights=None)(resize)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=final_activation, name="classification")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def wideresnet2810(input_shape, num_classes, final_activation, depth, width_factor, mod=True):
    n = (depth - 4) // 6
    num_filters = 16 * width_factor

    input_layer = Input(shape=input_shape)
    x = Conv2D(num_filters, kernel_size=(3, 3), strides=1, padding='same')(input_layer)
    x = BatchNormalization()(x)
    if mod:
        x = ReLU()(x)
    else:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Building residual blocks
    for _ in range(3):
        for _ in range(n):
            x = residual_block(x, num_filters, stride=1, mod=mod)
        num_filters *= 2  # Double the number of filters after each set of residual blocks

    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    output_layer = Dense(num_classes, activation=final_activation)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def inceptionv3(input_shape, num_classes, final_activation):
    inputs = tf.keras.layers.Input(shape=input_shape)
    resize = tf.keras.layers.UpSampling2D(size=(2, 2))(inputs)
    resize = tf.keras.layers.UpSampling2D(size=(2, 2))(resize)
    resize = tf.keras.layers.UpSampling2D(size=(2, 2))(resize)
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(256, 256, 3))(resize)

    #x = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    x = tf.keras.layers.Flatten()(base_model)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=final_activation, name="classification")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
    

def efficientnetb2(input_shape, num_classes, final_activation):    
    inputs = tf.keras.layers.Input(shape=input_shape)
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    base_model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, input_shape=(224,224,3), weights=None)

    base_output = base_model(resize)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_output)
    x = tf.keras.layers.BatchNormalization()(x)
    top_dropout_rate = 0.3
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=final_activation, name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
    
    
