## Tools to simulate Gaussian sources

A directory with examples is created by running `make examples`.

### gaussian_simulations



### uv_simulations



### uv_plots



### visualize_sampling
Creates visualization plots, which can be found in the examples directory.

Explanations:
* gaussian_source.pdf: A source consisting of Gaussian components. Can be one or two sided.
  The flux is decresing logarithmically towards the outer components.
* fft_gaussian_source.pdf: The Fourier transformation of the Gaussian source. Low frequencies 
  are located at the center, higher frequencies at the outer parts.
* uv_coverage.pdf: Simulated (uv)-coverage for a radio interferometric observation. The used antenna
  positions correspond to the layout of the [VLBA](https://science.nrao.edu/facilities/vlba/introduction-to-the-VLBA).
* baselines.gif/uv_coverage.gif: These gifs visualize the sampling during a radio interferometric observation.
* mask.pdf: 2d histogram of the (uv)-coverage. Masks are used to sample the Fourier space of Gaussian sources. By
  doing so, incomplete measurements of the sources are simulated.
* sampled_frequs.pdf: Visualizes the sampled frequencies of the source's Fourier space.
* recons_source.pdf: The inverse Fourier transformation of the incomplete sample.