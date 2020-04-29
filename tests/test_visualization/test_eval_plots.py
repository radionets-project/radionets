import os
from pathlib import Path

from tests.test_visualization.test_save_load_preds import test_create_h5_dataset

test_ds = test_create_h5_dataset()


def test_reshape_split():
    from gaussian_sources.inspection import reshape_split

    img_without = test_ds[0][1]
    img_with = test_ds[0][0].reshape(-1)

    assert reshape_split(img_without).shape == (64, 64)
    assert reshape_split(img_with)[0].shape == (64, 64)
    assert reshape_split(img_with)[1].shape == (64, 64)


def test_open_csv():
    from gaussian_sources.inspection import open_csv

    path = "tests/test_data/without_fourier/"
    mode = "predictions"

    assert open_csv(path, mode)[0].shape == (20, 4096)
    assert open_csv(path, mode)[1].shape == (20,)


def test_visualize_without_fourier():
    from gaussian_sources.inspection import visualize_without_fourier

    build = "tests/build/"
    build_diff = "tests/build/diff"
    build_blob = "tests/build/blob"
    Path(build_diff).mkdir(parents=True, exist_ok=True)
    Path(build_blob).mkdir(parents=True, exist_ok=True)

    img_input, img_pred, img_truth = (
        test_ds[0][0].reshape(-1),
        test_ds[1][1].reshape(-1),
        test_ds[2][1].reshape(-1),
    )
    i = 0

    assert visualize_without_fourier(i, img_input, img_pred, img_truth, build) is None


def test_visualize_with_fourier():
    from gaussian_sources.inspection import visualize_with_fourier

    img_input, img_pred, img_truth = (
        test_ds[0][0].reshape(-1),
        test_ds[1][0].reshape(-1),
        test_ds[2][0].reshape(-1),
    )
    i = 0
    build = "tests/build/"
    if os.path.exists(build) is False:
        os.mkdir(build)

    real_pred, imag_pred, real_truth, imag_truth = visualize_with_fourier(
        i, img_input, img_pred, img_truth, False, build
    )

    assert real_pred.shape == (64, 64)
    assert imag_pred.shape == (64, 64)
    assert real_truth.shape == (64, 64)
    assert imag_truth.shape == (64, 64)


def test_compute_fft():
    from gaussian_sources.inspection import visualize_fft, reshape_split

    img_pred, img_truth = (
        test_ds[0][0].numpy().reshape(-1),
        test_ds[1][0].numpy().reshape(-1),
    )
    real_pred, imag_pred = reshape_split(img_pred)
    real_truth, imag_truth = reshape_split(img_truth)
    i = 0
    build = "tests/build/"
    if os.path.exists(build) is False:
        os.mkdir(build)

    ifft_pred, ifft_truth = visualize_fft(
        i, real_pred, imag_pred, real_truth, imag_truth, False, build
    )

    assert ifft_pred.shape == (64, 64)
    assert ifft_truth.shape == (64, 64)

    return ifft_pred, ifft_truth


def test_hist_difference():
    from gaussian_sources.inspection import hist_difference

    img_pred, img_truth = (
        test_ds[0][1].numpy(),
        test_ds[1][1].numpy(),
    )
    build = "tests/build/"

    assert hist_difference(0, img_pred, img_truth, build) is None


def test_plot_difference():
    from gaussian_sources.inspection import plot_difference

    build = "tests/build/"
    ifft_pred, ifft_truth = test_compute_fft()
    img_pred, img_truth = (
        test_ds[0][1].numpy(),
        test_ds[1][1].numpy(),
    )
    dr_fourier = plot_difference(0, ifft_pred, ifft_truth, build)
    dr_wo_fourier = plot_difference(
        0, img_pred.reshape(64, 64), img_truth.reshape(64, 64), build
    )

    assert dr_fourier.dtype == float
    assert dr_wo_fourier.dtype == float


def test_blob_detection():
    from gaussian_sources.inspection import blob_detection

    img_pred, img_truth = (
        test_ds[0][1].numpy().reshape(64, 64),
        test_ds[9][1].numpy().reshape(64, 64),
    )
    build = "tests/build/"

    assert blob_detection(0, img_pred, img_truth, build) is None
