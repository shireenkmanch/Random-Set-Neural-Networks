# Random-Set Neural Networks (RS-NN)
[![arXiv](https://img.shields.io/badge/arXiv-2307.05772-b31b1b.svg)](https://arxiv.org/abs/2307.05772)

This repository contains code for the **ICLR 2025** paper: [Random-Set Neural Networks (RS-NN)](https://arxiv.org/abs/2307.05772).


## 📄 Abstract
Machine learning is increasingly deployed in safety-critical domains where erroneous predictions may lead to catastrophic consequences. This highlights the need for models to **know when they do not know**.  

In this paper, we propose a novel **Random-Set Neural Network (RS-NN)** approach for classification, which predicts **belief functions** instead of classical probability vectors. RS-NN leverages the mathematics of **random sets**—distributions over collections of sets of classes—to quantify **epistemic uncertainty** arising from limited or non-representative training data.  

RS-NN outperforms **Bayesian and Ensemble methods** in **accuracy, uncertainty estimation, and Out-of-Distribution (OoD) detection** across multiple benchmarks (CIFAR-10 vs SVHN/Intel-Image, MNIST vs FMNIST/KMNIST, ImageNet vs ImageNet-O). It also scales effectively to large architectures such as **WideResNet-28-10, VGG16, Inception V3, EfficientNetB2, and ViT-Base-16**, is robust to adversarial attacks, and can provide **statistical guarantees** in a conformal learning setting.

## 🚀 Getting Started

### **1️⃣ Environment Setup**
Set up the environment using `environment.yml`:
```
conda env create -f environment.yml
conda activate rsnn
```

For GPU support, ensure TensorFlow GPU version is installed:
```
python3 -m pip install tensorflow[and-cuda]
```

### **2️⃣ Training**
Train RS-NN using the **train.ipynb** notebook. Adjust the number of GPUs used during training with:
```
tf.config.experimental.set_visible_devices(gpus[:number_of_gpus], 'GPU')
```

### **3️⃣ Configuration**
Set the dataset, model architecture, and path for loading models in the train.ipynb notebook:
```
selected_dataset = "cifar10"  # Choose the dataset
selected_model = "resnet50"   # Choose the model
my_path = '/add/your/path/here'
```

For large-scale models, check the 📂*models* directory. For pre-trained models, run the *pretrained_models.py* file.

### **4️⃣ Evaluation**
Use the **eval.ipynb** notebook to 
✅ load trained models,
✅ evaluate **accuracy**, **uncertainty estimation**, and **Out-of-Distribution detection**.


## **📂 Repository Structure**
```
📦 RS-NN
├── 📁 baselines                             # Baseline models for comparison
│   └── 📁 ENN                               # Saved ENN predictions
│   └── 📁 LB-BNN                            # Saved LB-BNN predictions
├── 📁 data                                  # Dataloaders and datasets
│   └── 📁 datasets                          # Datasets for OoD detection: Intel Image, K-MNIST, SVHN
│   ├── adversarial_data.py                   # Code for adversarial dataset creation
│   ├── classes.py                            # List of classes for each dataset
│   ├── id_dataloader.py                      # In-distribution data loader
│   ├── ood_dataloader.py                     # Out-of-distribution data loader
├── 📁 metrics                               # Performance and evaluation metrics
│   ├── calibration_metrics.py                # Calibration error calculation metrics
│   ├── classification_metrics.py             # Classification accuracy metrics
│   ├── ood_metrics.py                        # Out-of-distribution detection metrics
│   ├── uncertainty_metrics.py                # Uncertainty quantification metrics
├── 📁 models                                # Model architectures and saved models
│   ├── models.py                             # Model definitions
│   ├── pretrained_models.py                  # Pre-trained model utilities
├── 📁 rsnn_functions                        # RSNN-specific functions
│   ├── belief_mass_betp.py                   # Belief mass calculation
│   ├── bf_encoding_gt.py                     # Encoding belief functions
│   ├── budgeting.py                          # Budgeting strategies
│   ├── rsnn_loss.py                          # Custom BCE loss functions
├── 📁 saved_models                          # Checkpoints and trained model weights
│   ├── CNN_resnet50_cifar10.h5               # CNN
│   ├── RSNN_resnet50_cifar10_weights.pkl     # RSNN
├── 📁 utils                                 # Utility scripts
│   ├── train_utils.py                        # Training utility functions
│   ├── eval_utils.py                         # Evaluation utility functions
├── 📄 .gitattributes                         # Git attributes configuration
├── 📄 README.md                              # This README file
├── 📜 environment.yml                        # Conda environment setup
├── 📄 eval.ipynb                             # Evaluation notebook
├── 📄 train.ipynb                            # Training notebook
├── 📄 new_classes.npy                        # Saved random set of classes
```



## **📢 Citation**
If you use this code, please cite our paper:

```
@inproceedings{
manchingal2025randomset,
title={Random-Set Neural Networks},
author={Shireen Kudukkil Manchingal and Muhammad Mubashar and Kaizheng Wang and Keivan Shariatmadar and Fabio Cuzzolin},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=pdjkikvCch}
}
```


## **📬 Contact**
For questions or issues, feel free to open an issue or contact the [author](shireenmohammed67@gmail.com).



## **⭐ Acknowledgments**
This research has received funding from the European Union’s Horizon 2020 Research and Innovation program under Grant Agreement No. 964505 (E-pi).
