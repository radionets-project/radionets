import re

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

import dl_framework.architectures as architecture
from dl_framework.data import get_bundles, h5_dataset
from dl_framework.model import load_pre_model
from mnist_cnn.visualize.utils import eval_model


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("pretrained_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.option("-num", type=int, required=False)
@click.option("-fourier", type=bool, required=True)
@click.option("-amp_phase", type=bool, required=True)
def main(data_path, arch, pretrained_path, out_path, fourier, amp_phase, num=20):
    """
    Create input, predictions and truth csv files for further investigation,
    such as visualize_predictions.

    Parameters
    ----------
    data_path : click path object
        path to the data files. Just the folder is necessary
    arch : string
        contains the name of the used architecture
    pretrained_path : click path object
        path to the pretrained model
    out_path : click path object
        path for the saving folder
    fourier : bool
        true, if the target images are fourier transformed
    num : int, optional
        number of images taken from the test dataset
    """
    bundle_paths = get_bundles(data_path)
    test = [path for path in bundle_paths if re.findall("fft_samp_test", path.name)]
    test_ds = h5_dataset(test, tar_fourier=fourier, amp_phase=amp_phase)
    indices = np.random.randint(0, len(test_ds), size=num)

    img_size = int(np.sqrt(test_ds[0][0].shape[1]))
    images = [test_ds[i][0].view(1, 2, img_size, img_size) for i in indices]
    images_x = [test_ds[i][0].numpy().reshape(-1) for i in indices]
    if fourier:
        images_y = [test_ds[i][1].numpy().reshape(-1) for i in indices]
    else:
        images_y = [test_ds[i][1].numpy().reshape(-1) for i in indices]

    arch = getattr(architecture, arch)()
    load_pre_model(arch, pretrained_path, visualize=True)

    prediction = [eval_model(img, arch).numpy().reshape(-1) for img in tqdm(images)]

    outpath = str(out_path) + "input.csv"
    df = pd.DataFrame(data=images_x, index=indices)
    df.to_csv(outpath, index=True)

    outpath = str(out_path) + "predictions.csv"
    df = pd.DataFrame(data=prediction, index=indices)
    df.to_csv(outpath, index=True)

    outpath = str(out_path) + "truth.csv"
    df_targets = pd.DataFrame(data=images_y, index=indices)
    df_targets.to_csv(outpath, index=True)


if __name__ == "__main__":
    main()
