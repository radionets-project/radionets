import os

from tests.test_save_load_preds import test_create_h5_dataset

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

    img_input, img_pred, img_truth = (
        test_ds[0][0].reshape(-1),
        test_ds[1][1].reshape(-1),
        test_ds[2][1].reshape(-1),
    )
    i = 0
    build = "tests/build/"
    os.mkdir(build)

    assert visualize_without_fourier(i, img_input, img_pred, img_truth, build) is None

    os.remove(build + "prediction_0.png")
    os.rmdir(build)


def test_visualize_with_fourier():
    from gaussian_sources.inspection import visualize_with_fourier

    img_input, img_pred, img_truth = (
        test_ds[0][0].reshape(-1),
        test_ds[1][0].reshape(-1),
        test_ds[2][0].reshape(-1),
    )
    i = 0
    build = "tests/build/"
    os.mkdir(build)

    real_pred, imag_pred, real_truth, imag_truth = visualize_with_fourier(
        i, img_input, img_pred, img_truth, build
    )

    assert real_pred.shape == (64, 64)
    assert imag_pred.shape == (64, 64)
    assert real_truth.shape == (64, 64)
    assert imag_truth.shape == (64, 64)

    os.remove(build + "prediction_0.png")
    os.rmdir(build)


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
    os.mkdir(build)

    assert visualize_fft(i, real_pred, imag_pred, real_truth, imag_truth, build) is None

    os.remove(build + "fft_pred_0.png")
    os.rmdir(build)
