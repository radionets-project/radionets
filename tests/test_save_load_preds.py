import re
import os
import numpy as np
import pandas as pd
from dl_framework.data import get_bundles, h5_dataset


def test_save_predictions():
    data_path = "tests/test_data/"
    fourier = False
    num = 3

    bundle_paths = get_bundles(data_path)
    test = [path for path in bundle_paths if re.findall("fft_samp_test", path.name)]
    test_ds = h5_dataset(test, tar_fourier=fourier)
    indices = np.random.randint(0, len(test_ds), size=num)

    assert len(indices) == 3
    assert int(np.sqrt(test_ds[0][0].shape[1])) == 64

    img_size = int(np.sqrt(test_ds[0][0].shape[1]))

    assert test_ds[0][0].view(1, 2, img_size, img_size).shape == (1, 2, 64, 64)
    assert test_ds[0][0].numpy().reshape(-1).shape == (8192,)
    assert test_ds[0][1].numpy().reshape(-1).shape == (4096,)

    test_imgs = [test_ds[0][0].numpy().reshape(-1), test_ds[1][1].numpy().reshape(-1)]
    build = "tests/build/"
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
    assert test_img[0].shape == (8192,)
    assert test_img[1].shape == (8192,)

    os.remove(path + mode + ".csv")
    os.rmdir(path)
