from pathlib import Path
from click.testing import CliRunner
import numpy as np
import traceback


def test_simulate_bundles():
    from gaussian_sources.simulate_bundles import main

    num_bundles_train = "1"
    num_bundles_valid = "1"
    num_bundles_test = "1"
    bundle_size = "5"
    img_size = "63"
    out_path = "./tests/build/gaussian_sources"
    Path(out_path).mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    options = [
        num_bundles_train,
        num_bundles_valid,
        num_bundles_test,
        bundle_size,
        img_size,
        out_path,
    ]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

    assert result.exit_code == 0


def test_create_fft_pairs_fourier():
    from gaussian_sources.create_fft_pairs import main

    data_path = "./tests/build/gaussian_sources"
    out_path = "./tests/build/gaussian_sources/fourier"
    Path(out_path).mkdir(parents=True, exist_ok=True)
    antenna_config = "./simulations/layouts/vlba.txt"

    runner = CliRunner()
    options = [
        data_path,
        out_path + "/fft_samp_",
        antenna_config,
        "-amp_phase",
        True,
        "-fourier",
        True,
        "-specific_mask",
        True,
        "-lon",
        -80,
        "-lat",
        50,
        "-steps",
        50,
        "-noise",
        True,
        "-preview",
        False,
    ]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

    assert result.exit_code == 0


def test_create_fft_pairs_wo_fourier():
    from gaussian_sources.create_fft_pairs import main

    data_path = "./tests/build/gaussian_sources"
    out_path = "./tests/build/gaussian_sources/wo_fourier"
    Path(out_path).mkdir(parents=True, exist_ok=True)
    antenna_config = "./simulations/layouts/vlba.txt"

    runner = CliRunner()
    options = [
        data_path,
        out_path + "/fft_samp_",
        antenna_config,
        "-amp_phase",
        False,
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
        "-noise",
        True,
        "-preview",
        False,
    ]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

    assert result.exit_code == 0


def test_normalization_fourier():
    from gaussian_sources.calculate_normalization import main
    import pandas as pd
    from dl_framework.data import get_bundles, open_fft_pair, do_normalisation
    import re
    import torch

    path = "./tests/build/gaussian_sources/fourier"

    data_path = path
    out_path = path + "/normalization_factors.csv"

    runner = CliRunner()
    options = [data_path, out_path]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

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
        path for path in bundle_paths if re.findall("fft_samp_train", path.name)
    ]

    bundles = [open_fft_pair(bund) for bund in bundle_paths]

    a = np.stack((bundles[0][0][:, 0], bundles[0][0][:, 1]), axis=1)

    assert np.isclose(do_normalisation(torch.tensor(a), factors).mean(), 0, atol=1e-1)
    assert np.isclose(do_normalisation(torch.tensor(a), factors).std(), 1, atol=1e-1)


def test_normalization_wo_fourier():
    from gaussian_sources.calculate_normalization import main
    import pandas as pd
    from dl_framework.data import get_bundles, open_fft_pair, do_normalisation
    import re
    import torch

    path = "./tests/build/gaussian_sources/wo_fourier"

    data_path = path
    out_path = path + "/normalization_factors.csv"

    runner = CliRunner()
    options = [data_path, out_path]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

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
        path for path in bundle_paths if re.findall("fft_samp_train", path.name)
    ]

    bundles = [open_fft_pair(bund) for bund in bundle_paths]

    a = np.stack((bundles[0][0][:, 0], bundles[0][0][:, 1]), axis=1)

    assert np.isclose(do_normalisation(torch.tensor(a), factors).mean(), 0, atol=1e-1)
    assert np.isclose(do_normalisation(torch.tensor(a), factors).std(), 1, atol=1e-1)


def test_train_cnn_fourier():
    from gaussian_sources.train_cnn import main

    data_path = "./tests/build/gaussian_sources/fourier"
    path_model = "./tests/build/gaussian_sources/fourier/test.model"
    arch = "UNet_fourier"
    norm_path = "./tests/build/gaussian_sources/fourier/normalization_factors.csv"
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
        True,
        "-amp_phase",
        True,
        "-pretrained",
        False,
        "-inspection",
        False,
        "-test",
        True,
    ]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

    assert result.exit_code == 0


def test_train_cnn_wo_fourier():
    from gaussian_sources.train_cnn import main

    data_path = "./tests/build/gaussian_sources/wo_fourier"
    path_model = "./tests/build/gaussian_sources/wo_fourier/test.model"
    arch = "UNet_denoise"
    norm_path = "./tests/build/gaussian_sources/wo_fourier/normalization_factors.csv"
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
        "-amp_phase",
        None,
        "-pretrained",
        False,
        "-inspection",
        False,
        "-test",
        True,
    ]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

    assert result.exit_code == 0


def test_find_lr():
    from gaussian_sources.find_lr import main

    data_path = "./tests/build/gaussian_sources/fourier"
    arch = "UNet_fourier"
    norm_path = "./tests/build/gaussian_sources/fourier/normalization_factors.csv"
    lr_type = "mse"

    runner = CliRunner()
    options = [
        data_path,
        arch,
        data_path,
        lr_type,
        norm_path,
        "-max_iter",
        "10",
        "-min_lr",
        "1e-6",
        "-max_lr",
        "1e-1",
        "-fourier",
        True,
        "-amp_phase",
        True,
        "-pretrained",
        False,
        "-save",
        True,
        "-test",
        True,
    ]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

    assert result.exit_code == 0
