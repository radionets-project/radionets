import click
import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from gaussian_sources.inspection import (
    visualize_without_fourier,
    visualize_with_fourier,
    visualize_fft,
    open_csv,
)


@click.command()
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.option("-fourier", type=bool, required=True)
@click.option("-log", type=bool, required=False)
@click.option("-index", type=int, required=False)
@click.option("-num", type=int, required=False)
def main(
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

    imgs_input, indices = open_csv(out_path, 'input')
    imgs_pred, _ = open_csv(out_path, 'predictions')
    imgs_truth, _ = open_csv(out_path, 'truth')

    if log is True:
        imgs_input = torch.log(imgs_input)

    if index is not None:
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        img_truth = imgs_truth[index]

    if index is None:
        print("\nPlotting {} pictures.\n".format(num))
        for i in tqdm(range(len(indices))):
            index = indices[i]
            img_input = imgs_input[i]
            img_pred = imgs_pred[i]
            img_truth = imgs_truth[i]
            if fourier:
                (
                    real_pred,
                    imag_pred,
                    real_truth,
                    imag_truth,
                ) = visualize_with_fourier(
                    i, img_input, img_pred, img_truth, out_path,
                )
                visualize_fft(i, real_pred, imag_pred, real_truth, imag_truth, out_path)
            else:
                visualize_without_fourier(
                    i, img_input, img_pred, img_truth, out_path
                )

    else:
        print("\nPlotting a single index.\n")
        i, index = 0, 0
        if fourier:
            real_pred, imag_pred, real_truth, imag_truth = visualize_with_fourier(
                i, img_input, img_pred, img_truth, out_path,
            )
            visualize_fft(i, real_pred, imag_pred, real_truth, imag_truth, out_path)
        else:
            visualize_without_fourier(
                i, img_input, img_pred, img_truth, out_path,
            )
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


if __name__ == "__main__":
    main()
