# radionets [![Build Status](https://travis-ci.com/Kevin2/radionets.svg?branch=master)](https://travis-ci.org/kevin2/radionets)

### Imaging radio interferometric data with neural networks

Executables to simulate and analyze radio interferometric data in python. The goal is to reconstruct (image) calibrated observations with convolutional neural networks. 
This repository is build up as a python package. After cloning you can install it with
    pip install .
While installing you may experience some problems with cartopy. In this case you have to install a proj and a geos library before:
`sudo apt-get -y install libgeos-dev`
`sudo apt-get -y install libproj-dev`
When you still have problems installing cartopy you can try the version on conda-forge:
`conda install --channel conda-forge cartopy`

At the moment the repository covers the following blocks:

## dl_framework

Framework used to create and train neural networks. Most of it can be found in [Practical Deep Learning for Coders, v3](https://course.fast.ai/index.html). Check it out for more information including a nice tutorial about deep learning.

## sampling

Functions to simulate and illustrate radio interferometric observations.

* Simulate antenna arrays
* Calcalute baselines
* Illustrate uv-coverages and baselines for different observations

## mnist_cnn

Feasibility study to test analysis strategies with convulotional neural networks.

* Reconstruct handwritten digits from their sampled Fourier spectrum
* Simulated VLBA observations used for sampling
* Simple CNN model for reconstruction and retransformation

## Versions used

* Python 3.7.5
* pyTorch 1.2.0
* cuda V10.1.243

A detailed view of all versions used can be found in setup.py.