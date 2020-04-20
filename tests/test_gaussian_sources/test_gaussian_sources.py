from pathlib import Path
from click.testing import CliRunner
import traceback


def test_simulate_bundles():
    from gaussian_sources.simulate_bundles import main

    num_bundles_train = "2"
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
        "-noise",
        True,
    ]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

    assert result.exit_code == 0


def test_create_fft_pairs():
    from gaussian_sources.create_fft_pairs import main

    data_path = "./tests/build/gaussian_sources"
    out_path = "./tests/build/gaussian_sources/fourier"
    Path(out_path).mkdir(parents=True, exist_ok=True)
    antenna_config = "./simulations/layouts/vlba.txt"

    runner = CliRunner()
    options = [
        data_path,
        out_path+"/fft_samp_",
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

    out_path = "./tests/build/gaussian_sources/wo_fourier"
    Path(out_path).mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    options = [
        data_path,
        out_path+"/fft_samp_",
        antenna_config,
        "-amp_phase",
        True,
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
