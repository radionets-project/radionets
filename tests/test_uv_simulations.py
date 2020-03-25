import numpy as np
'''
source
antenna

create mask
'''


def test_get_uv_coverage():
    from simulations.uv_simulations import (
        get_uv_coverage,
        get_antenna_config,
        source,
        antenna,
    )

    ant_config_path = './simulations/layouts/vlba.txt'
    s = source(120, 50)
    s.propagate(num_steps=20, multi_pointing=True)
    ant = antenna(*get_antenna_config(ant_config_path))

    u, v, steps = get_uv_coverage(s, ant)

    assert u.shape == v.shape
    assert len(u) / ant.baselines == steps
    assert len(v) / ant.baselines == steps


def test_sample_freqs():
    from simulations.uv_simulations import sample_freqs

    ant_config_path = './simulations/layouts/vlba.txt'
    img = np.ones((63, 63))
    img_samp = sample_freqs(img, ant_config_path)

    assert img_samp.shape == img.shape
    assert len(img[img == 1]) != len(img_samp[img_samp == 1])

    img_samp, mask = sample_freqs(img, ant_config_path, plot=True)

    assert img_samp.shape == img.shape
    assert mask.shape == img.shape


def test_get_antenna_config():
    from simulations.uv_simulations import get_antenna_config

    ant_config_path = './simulations/layouts/vlba.txt'
    ant_pos = get_antenna_config(ant_config_path)

    assert ant_pos.shape == (3, 10)
