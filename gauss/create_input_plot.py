import click
from tqdm import tqdm
from utils import open_mnist, add_noise


@click.command()
@click.argument('data_path', type=click.Path(exists=True, dir_okay=True))
def main(data_path):
    # Load MNIST dataset
    path = data_path
    train_x, valid_x = open_mnist(path)
    train_x = train_x[0:10]
    valid_x = valid_x[0:10]

    # Process train images, split into x and y
    i = 0
    for img in tqdm(train_x):
        a = add_noise(img, index=i, plotting=True)
        i = i + 1


if __name__ == '__main__':
    main()
