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
    out = adjust_outpath(path, "/test")

    assert type(path) == type(out)
    assert out.split("/")[-1] == "test0.h5"


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
        "-real_imag",
        True,
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

    assert (
        factors.keys()
        == ["train_mean_c0", "train_std_c0", "train_mean_c1", "train_std_c1", ]
    ).all()
    assert ~np.isnan(factors.values).all()
    assert ~np.isinf(factors.values).all()
    assert (factors.values != 0).all()

    bundle_paths = get_bundles(data_path)
    bundle_paths = [
        path for path in bundle_paths if re.findall("fft_bundle_samp_train", path.name)
    ]

    bundles = [open_fft_pair(bund) for bund in bundle_paths]

    a = np.stack((bundles[0][0][:, 0], bundles[0][0][:, 1]), axis=1)

    assert np.isclose(do_normalisation(torch.tensor(a), factors).mean(), 0, atol=1e-1)
    assert np.isclose(do_normalisation(torch.tensor(a), factors).std(), 1, atol=1e-1)


def test_train_cnn():
    from mnist_cnn.scripts.train_cnn import main

    data_path = "./tests/build"
    path_model = "./tests/build/test.model"
    arch = "UNet_denoise"
    norm_path = "./tests/build/normalization_factors.csv"
    epochs = "5"
    lr = "1e-3"
    lr_type = "mse"
    bs = "2"

    runner = CliRunner()
    options = [
        data_path,
        path_model,
        arch,
        norm_path,
        epochs,
        lr,
        lr_type,
        bs,
        "-fourier",
        False,
        "-pretrained",
        False,
        "-inspection",
        False,
        "-test",
        True,
    ]
    result = runner.invoke(main, options)

    assert result.exit_code == 0


def test_find_lr():
    from mnist_cnn.scripts.find_lr import main

    data_path = "./tests/build"
    arch = "UNet_denoise"
    norm_path = "./tests/build/normalization_factors.csv"
    lr_type = "mse"

    runner = CliRunner()
    options = [
        data_path,
        arch,
        data_path,
        lr_type,
        norm_path,
        "-max_iter",
        "400",
        "-min_lr",
        "1e-6",
        "-max_lr",
        "1e-1",
        "-fourier",
        False,
        "-pretrained",
        False,
        "-save",
        True,
    ]
    result = runner.invoke(main, options)

    assert result.exit_code == 0
