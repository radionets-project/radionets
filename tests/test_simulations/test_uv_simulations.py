import numpy as np


def test_source():
    from simulations.uv_simulations import source

    s = source(120, 50)

    assert s.lat == 50
    assert s.lon == 120
    assert len(s.to_ecef()) == 3
    assert len(s.propagate(num_steps=10)) == 2
    assert len(s.propagate(num_steps=10)[0]) == 20
    assert len(s.propagate(num_steps=10)[0]) == len(s.propagate(num_steps=10)[1])
    assert len(s.propagate(num_steps=10, multi_pointing=True)[0]) == 10


def test_antenna():
    from simulations.uv_simulations import (
        antenna,
        get_antenna_config,
    )

    ant = antenna(*get_antenna_config("simulations/layouts/vlba.txt"))

    assert ant.all.shape == (10, 3)
    assert ant.baselines == 90
    assert len(ant.X) & len(ant.Y) & len(ant.Z) == 10

    ant.to_geodetic(0, 0, 0)

    assert ant.lon is not None
    assert ant.lat is not None
    assert len(ant.to_enu(0, 0, 0)) == 2
    assert len(ant.to_enu(0, 0, 0)[0]) == 10
    assert len(ant.to_enu(0, 0, 0)[1]) == 10

    x_base, y_base = ant.get_baselines()

    assert len(x_base) & len(y_base) == 180

    u, v, steps = ant.get_uv()

    assert len(u) & len(v) == 90
    assert steps == 1


def test_uv_coverage():
    from simulations.uv_simulations import (
        get_uv_coverage,
        get_antenna_config,
        source,
        antenna,
        create_mask,
    )

    ant_config_path = "./simulations/layouts/vlba.txt"
    s = source(120, 50)
    s.propagate(num_steps=20, multi_pointing=True)
    ant = antenna(*get_antenna_config(ant_config_path))

    u, v, steps = get_uv_coverage(s, ant)

    assert u.shape == v.shape
    assert len(u) / ant.baselines == steps
    assert len(v) / ant.baselines == steps

    mask = create_mask(u, v)

    assert mask.shape == (63, 63)


def test_sample_freqs():
    from simulations.uv_simulations import sample_freqs

    ant_config_path = "./simulations/layouts/vlba.txt"
    img = np.ones((2, 63, 63))
    img_samp = sample_freqs(img, ant_config_path, test=True)

    assert img_samp.shape == img.shape
    assert len(img[img == 1]) != len(img_samp[img_samp == 1])

    img_samp, mask = sample_freqs(img, ant_config_path, plot=True, test=True)

    assert img_samp.shape == img.shape
    assert mask.shape == img.shape


def test_get_antenna_config():
    from simulations.uv_simulations import get_antenna_config

    ant_config_path = "./simulations/layouts/vlba.txt"
    ant_pos = get_antenna_config(ant_config_path)

    assert ant_pos.shape == (3, 10)
