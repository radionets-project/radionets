import click
import numpy as np
import pandas as pd

import dl_framework.architectures as architecture
from dl_framework.data import load_data
from dl_framework.inspection import eval_model
from dl_framework.model import load_pre_model


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("pretrained_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.option("-num", type=int, required=False)
@click.option("-fourier", type=bool, required=True)
def main(data_path, arch, pretrained_path, out_path, fourier, num=20):
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
    test_ds = load_data(data_path, "test", fourier=fourier)
    indices = np.random.randint(0, len(test_ds), size=num)

    images = test_ds[indices][0]
    images_x = test_ds[indices][0].numpy().reshape(num, -1)
    images_y = test_ds[indices][1].numpy().reshape(num, -1)

    arch = getattr(architecture, arch)()
    load_pre_model(arch, pretrained_path, visualize=True)

    prediction = eval_model(images, arch).numpy().reshape(num, -1)

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
