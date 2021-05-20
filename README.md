# Dementia Severity Classification under Small Sample Size and Weak Supervision in Thick Slice MRI

## Overview

Pytorch Implementation of the paper "Dementia Severity Classification under Small Sample Size and Weak Supervision in Thick Slice MRI". This project aims at the automatic classification of the Fazekas Scale of two visual biomarkers of dementia, namely, Periventricular White Matter (PVWM) changes and Deep White Matter (DWM) changes. When properly trained, the model is capable of assessing the severity of white matter lesions and their conlfuence in the brain MRI of a subject, hence accurately classifying the Fazekas Scale of the afformentioned visual biomarkers and providing insight into the progression of the disease and patient outcome.

This document describes how to train and use this model. A few input samples are provided give users the idea of how to organize the data and make sure that the training and inference stages work correctly. This document also describes the main functionality of the model. For a thourough discussion and greater details please consult the paper.

## Citation

If you use the code in this repository in your work, please cite the following paper:

XXXXXXX

## Table of Contents
* [1. Requirements and Installation ](#1-requirements-and-installation)
  * [1.1. Requirements](#11-requirements)
  * [1.2 Installation](#12-installation)
  * [1.3 Required Input Format](#13-required-input-format)
* [2. Running the Code](#2-running-the-code)
  * [2.1 Notebooks](#21-notebooks) 
  * [2.2 Training on a Small Dataset](#22-training-on-a-small-dataset)
  * [2.3 Running on GPU](#23-running-on-gpu)
* [3. Technical Details](#3-technical-details)
* [4. License](#4-license)
 
## 1. Requirements and Installation  

The network is written in Python. The deep learning framework is implemented using PyTorch to enable GPU acceleration for training/inference. It has been sucessfully tested on Ubuntu 18.04. You may also use Googla Colab and avoid installation trouble altogether!

### 1.1 Requirements

[Python 3.7](https://www.python.org/downloads/) and the following Python packages must be installed:
- [Pytorch](https://www.pytorch.org/): The open source Deep Learning framework.
- [Torchvision](https://pytorch.org/vision/stable/index.html): The python package that consists of popular datasets, model architectures, and common image transformations for computer vision.
- [Scikit-learn](https://scikit-learn.org/stable/): The Machine Learning library.
- [numpy](http://www.numpy.org/) : General purpose array-processing package.
- [Scikit-image](https://scikit-image.org/): The image processing library.
- [Pydicom](https://pydicom.github.io/): The python package for reading, modifying and writing DICOM data.
- [Matplotlib](https://matplotlib.org/): (Optional) The comprehensive library for creating visualizations in Python.
- [Cuda-toolkit](https://developer.nvidia.com/cuda-toolkit): (Optional) The CUDA Toolkit includes GPU-accelerated libraries, and the CUDA runtime for the Conda ecosystem.

### 1.2 Installation

The following instructions are for unix-like operating systems, but similar steps should also work for Windows.

Clone the repository using the command:
```
git clone https://github.com/dementia-classification/Dementia-Biomarkers-Classification/
```

Install [anaconda](https://www.anaconda.com/download) or [miniconda](https://conda.io/miniconda.html) and create a new virtual environment:

```
conda create -p ENV_FOLDER/my-conda-env python=3.7.0 -y
source activate ENV_FOLDER/my-conda-env
```

Then [Jupyterlab](https://jupyter.org/index.html) and all the required packages can be installed using Conda:
```
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-foge numpy scikit-learn matplotlib scikit-image pydicom
```

### 1.3 Required Input Format
The following must be noted:
- The provided code assumes all input and test data are in DICOM format. However, the code is easy to manipulate and change so that the model accepts as input a number of images rather than a DICOM file.
- Make sure that the ground-truth labels for training data are in a separate excel/csv file with columns including the fazekas scale of PVWM and DWM lesions of each subject.
- The structure of the training and test data should be similar to the sample data provided.


## 2. Running the Code

NOTE: First see [Section 1.2](#12-installation) for installation of the required packages.

### 2.1 Notebooks
The preprocessing jupyter notebook, the self-supervised learning jupyter notebook and the main jupyter notebook (for training the whole network and inference) are provided in the folder [dembio](https://github.com/dementia-classification/Dementia-Biomarkers-Classification/tree/main/dembio). It also contains a self_supervised.py file that contains the skeleton of the self-supervised learing component.  The notebooks must be run in the following order:

  1. The preprocessing notebook: It contains the code that reads the data of each MRI slice of the FLAIE MRI series of each subject from the "Dataset" directory, removes the background of each image and extracts the areas corresponding to  the white matter lesions. It stores the preprocessed form of each image in a new Directory called Preprocessed_Dataset.
  2. The self-supervised model pretraining notebook: it uses the original images as well as their preprocessed form to train a self-supervised network which is used as the initialization point for the feature extractor of the main network. The trained model is saved to the directory `../models/ss_models/self_supervised_model.pth.tar`.
  3. The main jupyter notebook: it uses the outputs (the preprocessed images and the trained self-supervised model) of the previous components and trains an instance of the whole model. It also contains the code that can be used to classify new data (inference). There is a cell at the begining of it that specifies all the configuration parameters for training and testing the model. 


### 2.2 Training on a Small Dataset

We have provided a few sample data in the [Dataset](https://github.com/dementia-classification/Dementia-Biomarkers-Classification/tree/main/data/Dataset) directory. By running the notebooks in the order mentioned in [Section 2.1](#21-notebooks), The preprocessed form of the data as well as a trained self-supervised classifier are created in the `Dataset` and the `models` directory, respectively. By running all cells of the main jupyter notebook, the final model is trained on a 10-fold cross validation setting. In each run the model is trained 120 epochs. The best models (with a name corresponding to the split and epoch it was attained) in each run of the algorithm are logged for later reference. By default these files are found in the `models/best_models/` directory. Although the location an be modified by changing the config instance.

Moreover, the training & validation metrics are also logged for visualisation via TensorBoard. Required log-files found at the `logidrs/runs` directory. See [Tensorboard Documentation](https://www.tensorflow.org/tensorboard/get_started) for further information. Use the following command to activate tensorboard:

```
tensorboard --logdir=./logdirs/runs
```

Finally, you can load each of the stored models and run it on unseen data.

### 2.3 Running on GPU

If you have access to cuda powered machine, simply set the value of the cuda attribute of the config instance to `True` and follow the instructions in the previous section. The processes should result in similar outputs as before.


## 3. Technical Details

Having briefly discussed the process of training the model on a small dataset, we now turn to the details in the training of the model. We also explain the main parameters that should be specified  in order for you to tailor the network and process to your needs. As previously stated the whole configuration parameters needed for instantiating and training the model are specified via an instance of the Class `Config` at the beginning of the main jupyter notebook. The code is well documented and the purpose of each config parameter is explained in the notebook. Regarldess, the following are worth noting:

- There are steps that read the MRI data into an array, and normalize the each input pixel to [0-255] range. Then the following image transformations are applied on each training image (The train and validation data transformations can be changed at will):
  - Make images grayscale
  - Random horizontal flips and random rotation in order to capture different patterns in MRI data acquistion in differenct centers.
  - Center crop the image to a predefined size. The parameter `image_crop_size` specifies this size.
- `label_type` defines the target of the classification task. It could be any of the values `PVWM` or `DWM` corresponding to the visual biomarkers. You can set it to any column name in your own labels csv file.
- The parameeters `pre_process` and `self_supervised` dictate the inclusion of each of the components in the whole network.
- The class `Config` also includes optimization parameters, training paramters such as the number of folds in the K-fold cross validation alogrithm, etc.

For further details regarding the architecture of the model please consult the paper.

## 4. License 

This software is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
