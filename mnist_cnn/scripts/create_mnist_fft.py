import click
from tqdm import tqdm
from mnist_cnn.scripts.utils import open_mnist, prepare_mnist_bundles


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.option("-size", type=int, required=True)
@click.option("-bundle_size", type=int, required=True)
@click.option("-noise", type=bool, required=False)
def main(data_path, out_path, size, bundle_size, noise=False):
    """
    Load MNIST dataset, split it into bundles, resize and compute fft.
    Bundles are saved to hdf5 files.

    Parameters
    ----------
    data_path: str
        path to MNIST pickle file
    out_path: str
        directory where bundles are saved

    Options
    -------
    size: int
        image size
    bundle_size: int
        number of images in one bundle
    noise: bool
        if true: images are noised before fft
    """
    train_x, valid_x = open_mnist(data_path)

    print("\nCreating train set.\n")
    bundles_train = train_x.reshape(
        int(len(train_x) / bundle_size), bundle_size, 28, 28
    )
    [
        prepare_mnist_bundles(bund, out_path, "train", pixel=size, noise=noise)
        for bund in tqdm(bundles_train)
    ]

    print("\nCreating valid set.\n")
    bundles_evaluation = valid_x.reshape(
        int(len(valid_x) / bundle_size), bundle_size, 28, 28
    )
    bundles_valid = bundles_evaluation[0:-10]
    [
        prepare_mnist_bundles(bund, out_path, "valid", pixel=size, noise=noise)
        for bund in tqdm(bundles_valid)
    ]

    print("\nCreating test set.\n")
    bundles_test = bundles_evaluation[-10:]
    [
        prepare_mnist_bundles(bund, out_path, "test", pixel=size, noise=noise)
        for bund in tqdm(bundles_test)
    ]


if __name__ == "__main__":
    main()
