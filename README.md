# radionets

### Imaging radio interferometric data with neural networks

Executables to simulate and analyze radio interferometric data in python. The goal is to reconstruct calibrated observations with convolutional neural networks. 
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

* Python 3.7.3
* pyTorch 1.2.0
* cuda V10.1.243
