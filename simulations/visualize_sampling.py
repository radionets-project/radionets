import matplotlib.pyplot as plt

from simulations.gaussian_simulations import gaussian_source
from simulations.uv_plots import (
    FT,
    plot_source,
    # animate_baselines,
    # animate_uv_coverage,
    plot_uv_coverage,
    apply_mask,
    plot_antenna_distribution,
    plot_mask,
)
from simulations.uv_simulations import (
    antenna,
    source,
    get_antenna_config,
    create_mask,
    get_uv_coverage,
)


sim_source = gaussian_source(63)

plot_source(sim_source, ft=False, log=True)
plt.savefig(
    "examples/gaussian_source.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05,
)

plot_source(sim_source, ft=True, log=True)
plt.savefig(
    "examples/fft_gaussian_source.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05,
)

ant = antenna(*get_antenna_config("./layouts/vlba.txt"))
s = source(-80, 40)
s.propagate()
# animate_baselines(s, ant, "examples/baselines", 5)
# animate_uv_coverage(s, ant, "examples/uv_coverage", 5)
s_lon = s.lon_prop
s_lat = s.lat_prop
plot_antenna_distribution(s_lon[0], s_lat[0], s, ant, baselines=True)
plt.savefig("examples/baselines.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)

u, v, steps = get_uv_coverage(s, ant, iterate=False)

fig = plt.figure(figsize=(6, 6), dpi=100)
plot_uv_coverage(u, v)
plt.ylim(-5e8, 5e8)
plt.xlim(-5e8, 5e8)
plt.savefig("examples/uv_coverage.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05)

fig = plt.figure(figsize=(8, 6), dpi=100)
mask = create_mask(u, v)
plot_mask(fig, mask)
plt.savefig("examples/mask.pdf", dpi=100, bbox_inches="tight", pad_inches=0.01)

sampled_freqs = apply_mask(FT(sim_source), mask)

plot_source(sampled_freqs, ft=False, log=True)
plt.xlabel("u", fontsize=20)
plt.ylabel("v", fontsize=20)
plt.savefig("examples/sampled_freqs.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05)

plot_source(sampled_freqs, ft=True, log=True)
plt.xlabel("l", fontsize=20)
plt.ylabel("m", fontsize=20)
plt.savefig("examples/recons_source.pdf", dpi=100, bbox_inches="tight", pad_inches=0.05)
