# radionets [![Actions Status](https://github.com/radionets-project/radionets/workflows/CI/badge.svg)](https://github.com/radionets-project/radionets/actions) [![codecov](https://codecov.io/gh/radionets-project/radionets/branch/main/graph/badge.svg)](https://codecov.io/gh/radionets-project/radionets)



## Imaging Radio Interferometric Data with Neural Networks

Executables to simulate and analyze radio interferometric data in python. The goal is to reconstruct calibrated observations with
convolutional Neural Networks to create high-resolution images. For further information, please have a look at our [paper](https://www.aanda.org/component/article?access=doi&doi=10.1051/0004-6361/202142113).

Analysis strategies leading to a reproducible processing and evaluation of data recorded by radio interferometers:
* Simulation of datasets (will be shifted to `vipy` repository)
* Training of deep learning models
* Reconstruction of radio interferometric data

## Installation (must be reworked after shifting simulation part to `vipy` repository)

This repository is build up as a python package. It is recommended to create an own conda environment to handle the dependencies of the packages.
You can create one by running the following command in this repository:
```
$ conda env create -f environment.yml
```
Depending on your `cuda` version you have to specify the `cudatoolkit` version used by `pytorch`. If you are working on machines
with `cuda` versions < 10.2, please change the version number in the environment.yml file. Since the package `pre-commit` is used, you need to execute 
```
$ pre-commit install
```
after the installation.

## Usage

For each task, executables are installed to your `PATH`. Each of them takes `toml` configuration files as input, to manage data paths and options.
Simulated data is saved in `hdf5`, trained models are saved as `pickle` files.

* `radionets_simulations <...>`
  This script is used to simulate radio interferometric datasets for the training of deep learning models.
* `radionets_training <...>`
  This script is used to train a model on events with known truth
  values for the target variable, usually monte carlo simulations.
* `radionets_evaluation <...>`
  This script is used to evaluate the performance of the trained deep learning models.

Default configuration files can be found in the examples directory. Additionally, the examples directory contains jupyter notebooks, which show an example
analysis pipeline and the corresponding commands.

## Structure of the Repository

### dl_framework

The used deep learning framework is based on [pytorch](https://pytorch.org/) and [fastai](https://www.fast.ai/).
An introduction to Neural Networks and an overview of the use of fastai to train deep learning models can be found in [Practical Deep Learning for Coders, v3](https://course.fast.ai/index.html) and [fastbook](https://github.com/fastai/fastbook).

### dl_training

Functions for handling the different training options. Currently, there are the training, the learning rate finder, and the loss plotting mode.

### simulations (further developed in [vipy](https://github.com/radionets-project/vipy) repository)

Functions to simulate and illustrate radio interferometric observations. At the moment simulations based on the MNIST dataset and 
simulations of Gaussian sources are possible. We are currently working on simulating visibilities directly in Fourier space.
For more information, visit our corresponding repository [vipy](https://github.com/radionets-project/vipy). In the future, the simulations will be created
using the `vipy` repository, while the `radionets` repository contains the training and evaluation methods.

### evaluation

Functions for the evaluation of the training sessions. The available options reach from single, exemplary plots in (u, v) space and image space to
methods computing characteristic values on large test datasets. In detail:

* Amplitude and phase for the prediction and the truth. Example image below including the difference between prediction and truth.
![](resources/amp_phase.png)
* Reconstructed source images with additional features, such as MS-SSIM values or the viewing angle. Example image below.
![](resources/source_plot.png)
* Histogram of differences between predicted and true viewing angles. Image includes a comparison with [wsclean](https://gitlab.com/aroffringa/wsclean).
![](resources/hist_jet_offsets.png)
* Histogram of ratio between predicted and true source areas. Image includes a comparison with [wsclean](https://gitlab.com/aroffringa/wsclean).
![](resources/hist_area_ratios.png)
* Histogram of flux difference in core component. Image includes a comparison with [wsclean](https://gitlab.com/aroffringa/wsclean).
![](resources/hist_mean_diffs.png)
* Included, but not yet fully operational
  * Histogram of differences between predicted and true MS-SSIM values on a dedicated test dataset
  * Histogram of differences between predicted and true dynamic range values on a dedicated test dataset

All histograms are created on a dedicated test dataset.

## Contributors

* Kevin Schmidt [@Kevin2](https://github.com/Kevin2)
* Felix Geyer [@FeGeyer](https://github.com/FeGeyer)
* Stefan Fröse [@StFroese](https://github.com/StFroese)
* Paul-Simon Blomenkamp [@PBlomenkamp](https://github.com/PBlomenkamp)
* Kevin Laudamus [@K-Lauda](https://github.com/K-Lauda)
* Emiliano Miranda [@emilianozm24](https://github.com/emilianozm24)
* Maximilian Büchel [@MaxBue](https://github.com/MaxBue)

## Versions used and tested

* Python >= 3.6
* pyTorch >= 1.2.0
* torchvision >= 0.4.0
* cudatoolkit >= 9.2
