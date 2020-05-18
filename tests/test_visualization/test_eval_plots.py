import os
from pathlib import Path


def test_load_data():
    from dl_framework.data import load_data

    data_path = "./tests/build/gaussian_sources/wo_fourier"
    test_ds = load_data(data_path, "test", False)
    return test_ds


def test_reshape_split():
    from gaussian_sources.inspection import reshape_split

    test_ds = test_load_data()

    img_without = test_ds[0][1]
    img_with = test_ds[0][0].reshape(-1)

    assert reshape_split(img_without).shape == (63, 63)
    assert reshape_split(img_with)[0].shape == (63, 63)
    assert reshape_split(img_with)[1].shape == (63, 63)


def test_visualize_without_fourier():
    from gaussian_sources.inspection import visualize_without_fourier

    test_ds = test_load_data()

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

    test_ds = test_load_data()

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

    assert real_pred.shape == (63, 63)
    assert imag_pred.shape == (63, 63)
    assert real_truth.shape == (63, 63)
    assert imag_truth.shape == (63, 63)


def test_compute_fft():
    from gaussian_sources.inspection import visualize_fft, reshape_split

    test_ds = test_load_data()

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

    assert ifft_pred.shape == (63, 63)
    assert ifft_truth.shape == (63, 63)

    return ifft_pred, ifft_truth


def test_hist_difference():
    from gaussian_sources.inspection import hist_difference

    test_ds = test_load_data()

    img_pred, img_truth = (
        test_ds[0][1].numpy(),
        test_ds[1][1].numpy(),
    )
    build = "tests/build/"

    assert hist_difference(0, img_pred, img_truth, build) is None


def test_plot_difference():
    from gaussian_sources.inspection import plot_difference

    test_ds = test_load_data()

    build = "tests/build/"
    ifft_pred, ifft_truth = test_compute_fft()
    img_pred, img_truth = (
        test_ds[0][1].numpy(),
        test_ds[1][1].numpy(),
    )
    dr_fourier = plot_difference(0, ifft_pred, ifft_truth, 1e-6, build)
    dr_wo_fourier = plot_difference(
        0, img_pred.reshape(63, 63), img_truth.reshape(63, 63), 1e-6, build
    )

    assert dr_fourier.dtype == float
    assert dr_wo_fourier.dtype == float


def test_blob_detection():
    from gaussian_sources.inspection import blob_detection

    test_ds = test_load_data()

    img_pred, img_truth = (
        test_ds[0][1].numpy().reshape(63, 63),
        test_ds[3][1].numpy().reshape(63, 63),
    )
    build = "tests/build/"

    assert blob_detection(0, img_pred, img_truth, build) is None
