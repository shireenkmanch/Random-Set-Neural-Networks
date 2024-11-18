
import tensorflow as tf
import pickle
from rsnn_functions.rsnn_loss import BinaryCrossEntropy
from rsnn_functions.belief_mass_betp import mass_coeff
import numpy as np
import keras

@keras.saving.register_keras_serializable(package="my_package", name="BinaryCrossEntropy")

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
    - model_type: A string specifying the model type ('CNN' or 'RSNN').

    Returns:
    A loaded Keras model.
    """
    if model_type not in ['CNN', 'RSNN']:
        raise ValueError("Invalid model type. Supported types are 'CNN' and 'RSNN'.")

    model_name = f'{model_type}_{selected_model}_{selected_dataset}'

    custom_objects = {}

    if model_type == 'RSNN':
        new_classes = np.load('new_classes.npy', allow_pickle=True)
        mass_coeff_matrix = mass_coeff(new_classes)
        mass_coeff_matrix = tf.cast(mass_coeff_matrix, tf.float32)
        with open(f'saved_models/{model_name}_weights.pkl', 'rb') as weights_file:
            saved_weights = pickle.load(weights_file)
        # loaded_model = tf.keras.models.load_model(f'saved_models/{model_name}.keras', custom_objects={'BinaryCrossEntropy': BinaryCrossEntropy})

        # Set weights
        inputs = tf.keras.layers.Input(shape=(32, 32, 3))
        resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
        new_base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights=None)(resize)
        x = tf.keras.layers.GlobalAveragePooling2D()(new_base_model)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        outputs = tf.keras.layers.Dense(len(new_classes), activation="sigmoid", name="classification")(x)

        loaded_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        loaded_model.compile(loss=BinaryCrossEntropy,
                    optimizer="adam",
                    metrics=['binary_accuracy'])
        loaded_model.summary()
        loaded_model.set_weights(saved_weights)

    elif model_type == 'CNN':
        loaded_model = tf.keras.models.load_model(f'saved_models/{model_name}.h5')

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
