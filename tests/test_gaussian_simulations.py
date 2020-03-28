import numpy as np
import os
from unittest.mock import patch


def test_create_rot_mat():
    from simulations.gaussian_simulations import create_rot_mat

    rot_mat = create_rot_mat(np.deg2rad(20))

    assert rot_mat.shape == (2, 2)


def test_create_grid():
    from simulations.gaussian_simulations import create_grid

    grid = create_grid(63)

    assert grid.shape == (3, 63, 63)
    assert grid[0].min() == 1e-10


def test_gauss_parameters():
    from simulations.gaussian_simulations import gauss_paramters

    params = gauss_paramters()

    assert len(params) == 8
    assert type(params[0]) == int
    assert type(params[6]) == int
    assert type(params[7]) == int
    assert len(params[1]) == params[0]
    assert len(params[2]) == params[0]
    assert len(params[3]) == params[0]
    assert len(params[4]) == params[0]
    assert len(params[5]) == params[0]


def test_gaussians():
    from simulations.gaussian_simulations import (
        create_grid,
        gauss_paramters,
        add_gaussian,
        gaussian_component,
        create_gaussian_source,
        create_bundle,
        create_n_bundles,
    )

    img = create_grid(63)
    img_gauss = add_gaussian(img, amp=1, x=0, y=0, sig_x=1, sig_y=1, rot=0)

    assert img_gauss.shape == img[0].shape
    assert np.isclose(img_gauss.max(), 1)
    assert np.where(np.isclose(img_gauss, 1)) == (img.shape[1] // 2, img.shape[2] // 2)

    comp = gaussian_component(img[1], img[2], 1, 1, 1)

    assert comp.shape == img[0].shape
    assert np.isclose(comp.max(), 1)
    assert np.where(np.isclose(comp, 1)) == (img.shape[1] // 2, img.shape[2] // 2)

    params = gauss_paramters()
    source = create_gaussian_source(img, *params)

    assert source.shape == img[0].shape

    bundle = create_bundle(63, 10)

    assert bundle.shape == (10, 63, 63)

    path = "./tests/build/bundle_"
    os.mkdir("./tests/build")
    create_n_bundles(1, 10, 63, path)

    assert os.path.exists(path + "0.h5")

    os.remove(path + "0.h5")
    os.rmdir("./tests/build")


@patch("simulations.gaussian_simulations.plt.show")
def test_noise(mock_show):
    from simulations.gaussian_simulations import add_noise, get_noise

    img = np.ones((63, 63))
    bundle = np.ones((2, 63, 63))

    noise = get_noise(img, scale=0.05)

    assert noise.shape == img.shape

    noisy_bundle = add_noise(bundle)

    assert noisy_bundle.shape == bundle.shape

    assert add_noise(bundle, preview=True).shape == bundle.shape
