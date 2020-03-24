import numpy as np
'''
source
antenna

get uc coverage

create mask
'''


def test_sample_freqs():
    from simulations.uv_simulations import sample_freqs

    config_path = './simulations/layouts/vlba.txt'
    img = np.ones((63, 63))
    img_samp = sample_freqs(img, config_path)

    assert img_samp.shape == img.shape
    assert len(img[img == 1]) != len(img_samp[img_samp == 1])

    img_samp, mask = sample_freqs(img, config_path, plot=True)

    assert img_samp.shape == img.shape
    assert mask.shape == img.shape


def test_get_antenna_config():
    from simulations.uv_simulations import get_antenna_config

    path = './simulations/layouts/vlba.txt'
    ant_pos = get_antenna_config(path)

    assert ant_pos.shape == (3, 10)
