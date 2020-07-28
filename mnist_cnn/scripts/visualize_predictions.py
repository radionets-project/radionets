import click
from pathlib import Path
from dl_framework.data import load_data
from mnist_cnn.scripts.visualize import plot_results
from dl_framework.inspection import (
    load_pretrained_model,
    get_images,
    eval_model,
    reshape_2d,
)


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("arch_name", type=str)
@click.argument("model_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("norm_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-num_images", type=int, default=5, required=False, help="Disable logger in tests",
)
def main(data_path, arch_name, model_path, norm_path, num_images):
    """
    Visualize predictions of trained model.
    Creates n plots containing input, prediction and traget image.

    Parameters
    ----------
    data_path: str
        path to data directory
    arch_name: str
        name of architecture
    model_path: str
        path to saved model
    norm_path: str
        path to normalization factors

    Options
    -------
    num_images: int
        number of plotted images, default is 5
    """
    # Load data and create train and valid datasets
    test_ds = load_data(data_path, "test", fourier=False)

    arch = load_pretrained_model(arch_name, model_path)

    img_test, img_true = get_images(test_ds, num_images, norm_path)

    pred = eval_model(img_test, arch)

    out_path = Path(model_path).parent

    plot_results(img_test, reshape_2d(pred), reshape_2d(img_true), out_path, save=True)


if __name__ == "__main__":
    main()
