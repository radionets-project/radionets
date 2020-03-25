import numpy as np
import os


def test_plot_uv_coverage():
    from simulations.uv_plots import plot_uv_coverage

    u = np.array(range(5))
    v = np.array(range(5))

    assert plot_uv_coverage(u, v) is None


def test_plot_baselines():
    from simulations.uv_plots import plot_baselines
    from simulations.uv_simulations import (
        antenna,
        get_antenna_config,
    )

    ant_config_path = "simulations/layouts/vlba.txt"
    ant = antenna(*get_antenna_config(ant_config_path))
    ant.to_enu(0, 0, 0)

    assert plot_baselines(ant) is None


def test_plot_antenna_distribution():
    from simulations.uv_plots import plot_antenna_distribution
    from simulations.uv_simulations import (
        antenna,
        source,
        get_antenna_config,
    )

    source_lon = 120
    source_lat = 50
    s = source(120, 50)
    ant_config_path = "simulations/layouts/vlba.txt"
    ant = antenna(*get_antenna_config(ant_config_path))

    assert plot_antenna_distribution(source_lon, source_lat, s, ant) is None
    assert (
        plot_antenna_distribution(source_lon, source_lat, s, ant, baselines=True)
        is None
    )


def test_animate():
    from simulations.uv_plots import (
        animate_baselines,
        animate_uv_coverage,
    )
    from simulations.uv_simulations import (
        antenna,
        source,
        get_antenna_config,
    )

    build = "./tests/build"
    baseline = "./tests/build/baseline"
    uv_coverage = "./tests/build/uv_coverage"
    s = source(120, 50)
    s.propagate()
    ant_config_path = "simulations/layouts/vlba.txt"
    ant = antenna(*get_antenna_config(ant_config_path))
    os.mkdir(build)

    assert animate_baselines(s, ant, baseline) is None
    assert animate_uv_coverage(s, ant, uv_coverage) is None

    os.remove(baseline + ".gif")
    os.remove(uv_coverage + ".gif")
    os.rmdir("./tests/build")


def test_plot_source():
    from simulations.uv_plots import plot_source

    img = np.ones((5, 5))

    assert plot_source(img) is None
    assert plot_source(img, ft=True) is None
    assert plot_source(img, ft=True, log=True) is None


def test_FT():
    from simulations.uv_plots import FT

    img = np.ones((5, 5))
    ft = FT(img)

    assert img.shape == ft.shape


def test_apply_mask():
    from simulations.uv_plots import apply_mask

    img = np.ones((5, 5))
    mask = np.ones((5, 5))

    img_masked = apply_mask(img, mask)

    assert img.shape == img_masked.shape
