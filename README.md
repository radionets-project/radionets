# radionets [![Build Status](https://travis-ci.com/Kevin2/radionets.svg?branch=master)](https://travis-ci.org/kevin2/radionets)

## Imaging Radio Interferometric Data with Neural Networks

Executables to simulate and analyze radio interferometric data in python. The goal is to reconstruct calibrated observations with convolutional neural networks to create high resolution images. 
This repository is build up as a python package. It is recommended to create an own conda environment to handle the dependencies of the packages. You can create one by running the folliwing command in this repository:
```
$ conda env create -f environment.yaml
```
Depending on your cuda version you have to specify the cudatoolkit version used by pytorch. If you working on machines with cuda versions < 10.2, please
change the version number in the environment.yml file.

## Usage




## Structure of the Repository

### dl_framework

The used deep learning framework is based in [pytorch](https://pytorch.org/) and [fastai](https://www.fast.ai/).
'''
explain structure, idea and functionality
'''

Framework used to create and train neural networks. Most of it can be found in [Practical Deep Learning for Coders, v3](https://course.fast.ai/index.html). Check it out for more information including a nice tutorial about deep learning.

### simulations

'''
At the end one simulation block with all kind of sources: MNIST and Gaussian sources, which are split into point, pointlike and extended

overview of settings and possibilities
details and example images in simulations directory
example to run the code
'''

Functions to simulate and illustrate radio interferometric observations.

* Define antenna arrays
* Calculate baselines
* Simulate (uv)-coverages
* Create (uv)-masks
* Illustrate uv-coverages and baselines for different observations

### mnist_cnn

Feasibility study to test analysis strategies with convolutional neural networks.

* Reconstruct handwritten digits from their sampled Fourier spectrum
* Simulated VLBA observations used for sampling
* Simple CNN model for reconstruction and retransformation

All analysis steps can be run using the Makefile inside the mnist_cnn directory.
The different steps for an example analysis are:
1. mnist_fft: rescale and create the Fourier transformation of the mnist images
2. mnist_samp: sample the Fourier space with simulated (uv)-coverages
3. calc_normalization: calculate normalization factors to normalize train and valid dataset
4. cnn_training: train the convolutional neural network, many options are available here

### Evaluation

'''
evaluate training process with different approaches

ms-ssim
blob-detection
'''

## Versions used and tested

* Python >= 3.6
* pyTorch >= 1.2.0
* torchvision >= 0.4.0
* cudatoolkit >= 9.2