import click
from dl_framework.data import load_data
from mnist_cnn.scripts.visualize import plot_dataset


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.option(
    "-num_images",
    type=int,
    default=4,
    required=False,
    help="Disable logger in tests",
)
def main(data_path, num_images):
    # Load data and create train and valid datasets
    train_ds = load_data(data_path, "train", fourier=False)

    plot_dataset(train_ds, num_images, save=True)


if __name__ == "__main__":
    main()
