import click
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import dl_framework.architectures as architecture
from dl_framework.model import load_pre_model
from dl_framework.data import get_bundles, h5_dataset
from tqdm import tqdm
import re
from gaussian_sources.inspection import (
    visualize_without_fourier,
    visualize_with_fourier,
    visualize_fft,
)


@click.command()
@click.argument("arch", type=str)
@click.argument("pretrained_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("data_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.option("-fourier", type=bool, required=True)
@click.option("-log", type=bool, required=False)
@click.option("-index", type=int, required=False)
@click.option("-num", type=int, required=False)
def main(
    arch,
    pretrained_path,
    data_path,
    out_path,
    fourier,
    index=None,
    log=False,
    num=None,
):
    # to prevent the localhost error from happening
    # first change the backende and second turn off
    # the interactive mode
    matplotlib.use("Agg")
    plt.ioff()
    plt.rcParams.update({"figure.max_open_warning": 0})
    bundle_paths = get_bundles(data_path)
    test = [path for path in bundle_paths if re.findall("fft_samp_test", path.name)]
    test_ds = h5_dataset(test, tar_fourier=fourier)

    if index is None:
        indices = np.random.randint(0, len(test_ds), size=num)
        img = [test_ds[i][0] for i in indices]
        img_y = [test_ds[i][1] for i in indices]
    else:
        img = [test_ds[index][0]]
        img_y = [test_ds[index][1]]

    if log is True:
        img = torch.log(img)

    # get arch
    arch = getattr(architecture, arch)()

    # load pretrained model
    load_pre_model(arch, pretrained_path, visualize=True)

    if index is None:
        print("\nPlotting {} pictures.\n".format(num))
        for i in tqdm(range(len(indices))):
            index = indices[i]
            if fourier:
                (
                    real_pred,
                    imag_pred,
                    real_truth,
                    imag_truth,
                ) = visualize_with_fourier(i, img, img_y, arch, out_path,)
                visualize_fft(i, real_pred, imag_pred, real_truth, imag_truth, out_path)
            else:
                visualize_without_fourier(i, index, img, img_y, arch, out_path)

    else:
        print("\nPlotting a single index.\n")
        i, index = 0, 0
        if fourier:
            real_pred, imag_pred, real_truth, imag_truth = visualize_with_fourier(
                i, img, img_y, arch, out_path
            )
            visualize_fft(i, real_pred, imag_pred, real_truth, imag_truth, out_path)
        else:
            visualize_without_fourier(i, index, img, img_y, arch, out_path)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


if __name__ == "__main__":
    main()
