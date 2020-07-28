import os
import numpy as np
import pandas as pd
from dl_framework.data import load_data


def test_create_h5_dataset():
    data_path = "./tests/build/gaussian_sources/wo_fourier"
    fourier = False

    test_ds = load_data(data_path, "test", fourier=fourier)

    img = test_ds[0][0]
    img_y = test_ds[0][1]

    assert img[0].shape == (63, 63)
    assert img[1].shape == (63, 63)
    assert img_y.shape == (3969,)

    return test_ds


def test_save_predictions():
    num = 3

    test_ds = test_create_h5_dataset()
    indices = np.random.randint(0, len(test_ds), size=num)

    assert len(indices) == 3
    assert test_ds[0][0].shape[1] == 63

    img_size = test_ds[0][0].shape[1]

    assert test_ds[0][0].view(1, 2, img_size, img_size).shape == (1, 2, 63, 63)
    assert test_ds[0][0].numpy().reshape(-1).shape == (7938,)
    assert test_ds[0][1].numpy().reshape(-1).shape == (3969,)

    test_imgs = [test_ds[0][0].numpy().reshape(-1), test_ds[1][1].numpy().reshape(-1)]
    build = "tests/build/"
    if os.path.exists(build) is False:
        os.mkdir(build)

    outpath = build + "input.csv"
    df = pd.DataFrame(data=test_imgs, index=[1, 2])
    df.to_csv(outpath, index=True)


def test_load_predictions():
    from gaussian_sources.inspection import open_csv

    path = "tests/build/"
    mode = "input"
    test_img, indices = open_csv(path, mode)

    assert indices[0] == 1
    assert indices[1] == 2
    assert test_img[0].shape == (7938,)
    assert test_img[1].shape == (7938,)
