# radionets [![Build Status](https://travis-ci.com/Kevin2/radionets.svg?branch=master)](https://travis-ci.org/kevin2/radionets)

### Imaging radio interferometric data with neural networks

Executables to simulate and analyze radio interferometric data in python. The goal is to reconstruct (image) calibrated observations with convolutional neural networks. 
This repository is build up as a python package. After cloning you can install it with
`pip install .` after navigating to the folder. We advise you to create a clean conda environment first.
While installing you may experience some problems with cartopy as it has dependencies with geos and proj which can not be handled by pip.
To successfully install cartopy you have to install geos and proj with conda:
```
conda install geos pyproj
```

At the moment the repository covers the following blocks:

## dl_framework

'''
explain structure, idea and functionality
'''

Framework used to create and train neural networks. Most of it can be found in [Practical Deep Learning for Coders, v3](https://course.fast.ai/index.html). Check it out for more information including a nice tutorial about deep learning.

## simulations

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

## mnist_cnn

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

## Evaluation

'''
evaluate training process with different approaches

ms-ssim
blob-detection
'''

## Versions used and tested

* Python 3.7.5
* pyTorch 1.2.0
* torchvision 0.4.0
* cuda V10.1.243

* Python 3.7.5
* pyTorch 1.5.0
* torchvision 0.6.0
* cuda V10.1.243

A detailed view of all versions used can be found in setup.py.