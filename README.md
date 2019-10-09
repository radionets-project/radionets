# radionets

### Imaging radio interferometric data with neural networks.

Executables to simulate and analyze radio interferometric data in python. The goal is to reconstruct calibrated observations with convolutional neural networks. 
The different directories of the repository form modules to perform individual analysis steps. You can find tools to simulate radio interferometric observations and a deep learning training framework, which is highly inspired by the fast.ai v3 course. ([Practical Deep Learning for Coders, v3](https://course.fast.ai/index.html))
The mnist_cnn directory contains a first feasibility study using toy data created with the mnist dataset.

## Sampling

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
* pyTorch 1.1.0
* cuda V10.0.130
