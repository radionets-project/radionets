import numpy as np
import os
from click.testing import CliRunner


def test_open_mnist():
    from mnist_cnn.scripts.utils import open_mnist

    path = "./resources/mnist.pkl.gz"
    x_train, x_valid = open_mnist(path)
    assert x_train.shape == (50000, 784)
    assert x_valid.shape == (10000, 784)


def test_adjust_outpath():
    from mnist_cnn.scripts.utils import adjust_outpath

    path = "this/is/a/path"
    out = adjust_outpath(path, "test")

    assert type(path) == type(out)
    assert out.split("/")[-1] == "fft_bundle_test0.h5"


def test_prepare_mnist_bundles():
    from mnist_cnn.scripts.utils import prepare_mnist_bundles

    bundle = np.ones((10, 3, 3))
    build = "./tests/build"
    if os.path.exists(build) is False:
        os.mkdir(build)

    assert prepare_mnist_bundles(bundle, build, "test", noise=True, pixel=5) is None


def test_create_mnist_fft():
    from mnist_cnn.scripts.create_mnist_fft import main

    data_path = "./resources/mnist_test.pkl.gz"
    out_path = "./tests/build"

    if os.path.exists(out_path) is False:
        os.mkdir(out_path)

    runner = CliRunner()
    options = [data_path, out_path, "-size", 63, "-bundle_size", 2]
    result = runner.invoke(main, options)

    assert result.exit_code == 0


def test_create_fft_sampled():
    from mnist_cnn.scripts.create_fft_sampled import main

    data_path = "./tests/build/"
    out_path = "./tests/build"
    antenna_config = "./simulations/layouts/vlba.txt"

    runner = CliRunner()
    options = [
        data_path,
        out_path,
        antenna_config,
        "-fourier",
        False,
        "-specific_mask",
        True,
        "-lon",
        -80,
        "-lat",
        50,
        "-steps",
        50,
    ]
    result = runner.invoke(main, options)

    assert result.exit_code == 0


def test_normalization():
    from mnist_cnn.scripts.calculate_normalization import main
    import pandas as pd
    from dl_framework.data import get_bundles, open_fft_pair, do_normalisation
    import re
    import torch

    data_path = "./tests/build"
    out_path = "./tests/build/normalization_factors.csv"

    runner = CliRunner()
    options = [data_path, out_path]
    result = runner.invoke(main, options)

    assert result.exit_code == 0

    factors = pd.read_csv(out_path)

    assert (factors.keys() == [
        "train_mean_real",
        "train_std_real",
        "train_mean_imag",
        "train_std_imag",
    ]).all()
    assert ~np.isnan(factors.values).all()
    assert ~np.isinf(factors.values).all()
    assert (factors.values != 0).all()

    bundle_paths = get_bundles(data_path)
    bundle_paths = [
        path for path in bundle_paths if re.findall("fft_bundle_samp_train", path.name)
    ]

    bundles = [open_fft_pair(bund) for bund in bundle_paths]

    a = np.stack((bundles[0][0].real, bundles[0][0].imag), axis=1)

    assert np.isclose(do_normalisation(torch.tensor(a), factors).mean(), 0, atol=1e-1)
    assert np.isclose(do_normalisation(torch.tensor(a), factors).std(), 1, atol=1e-1)
