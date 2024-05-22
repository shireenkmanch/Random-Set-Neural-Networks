# Random_Set_Neural_Networks (RS-NN)

This repository contains code for Random-Set Neural Networks (RS-NN).

## Environment Setup

Set up the environment using `environment.yml`:
```
conda env create -f environment.yml
   conda activate myenv
```

For GPU support, ensure TensorFlow GPU version is installed:
```
python3 -m pip install tensorflow[and-cuda]
```

## Training
For training, run the **train.ipynb** notebook. Adjust the number of GPUs used during training with:
```
tf.config.experimental.set_visible_devices(gpus[:number_of_gpus], 'GPU')
```

## Configuration
Set the dataset, model architecture, and path for loading models in the train.ipynb notebook:
```
selected_dataset = "cifar10"  # Choose the dataset
selected_model = "resnet50"   # Choose the model
my_path = '/add/your/path/here'
```

For large-scale models, check the models directory. For pre-trained models, check the pretrained_models directory.

## Evaluation
Use the **eval.ipynb** notebook to load trained models and view experimental results on Out-of-distribution detection, accuracy, uncertainty estimation, etc.




