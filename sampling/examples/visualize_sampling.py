import sys
sys.path.append('..')
import numpy as np
from source_simulations import simulate_gaussian_source
import matplotlib.pyplot as plt

from uv_plots import FT, plot_image, plot_frequencies, animate_baselines, animate_uv_coverage, plot_uv_coverage, apply_mask
from uv_simulations import antenna, source, get_antenna_config, create_mask, get_uv_coverage

from obspy.imaging.cm import viridis_white_r



gaussian_source = simulate_gaussian_source(64, blur=True)

plot_image(gaussian_source)
plt.savefig("gaussian_source.pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

plot_frequencies(gaussian_source)
plt.savefig("fft_gaussian_source.pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

ant = antenna(*get_antenna_config('../layouts/vlba.txt'))
s = source(-80, 40)
s.propagate()
animate_baselines(s, ant, 'baselines', 5)
animate_uv_coverage(s, ant, 'uv_coverage', 5)

u, v, steps = get_uv_coverage(s, ant, iterate=False)

fig = plt.figure(figsize=(6, 6), dpi=100)
plot_uv_coverage(u, v)
plt.ylim(-5e8, 5e8)
plt.xlim(-5e8, 5e8)
plt.savefig("uv_coverage.pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(111)
mask = create_mask(u, v)
plt.imshow(mask, cmap=viridis_white_r)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')
plt.savefig("mask.pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

sampled_freqs = apply_mask(FT(gaussian_source), mask)

plot_frequencies(sampled_freqs, ft=False)
plt.savefig("sampled_freqs.pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

plot_image(sampled_freqs, ft=True)
plt.savefig("recons_source.pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)