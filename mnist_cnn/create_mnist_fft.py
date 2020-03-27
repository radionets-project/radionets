import click
from tqdm import tqdm
from utils import open_mnist, prepare_mnist_bundles


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.option("-noise", type=bool, required=False)
@click.option("-size", type=int, required=True)
def main(data_path, out_path, size, noise=False):
    """
    Load MNIST dataset, split it into bundles, resize and compute fft.
    Bundles are saved to hdf5 files.

    Parameters
    ----------
    data_path: str
        path to MNIST pickle file
    out_path: str
        directory where bundles are saved
    noise: bool
        if true: images are noised before fft
    """
    path = data_path
    train_x, valid_x = open_mnist(path)

    bundles_train = train_x.reshape(100, 500, 28, 28)
    [
        prepare_mnist_bundles(bund, out_path, "train", pixel=size, noise=noise)
        for bund in tqdm(bundles_train[0:1])
    ]

    bundles_valid = valid_x.reshape(20, 500, 28, 28)
    [
        prepare_mnist_bundles(bund, out_path, "valid", pixel=size, noise=noise)
        for bund in tqdm(bundles_valid[0:1])
    ]


if __name__ == "__main__":
    main()
