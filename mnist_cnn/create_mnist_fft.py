import click
import numpy as np
from tqdm import tqdm
from utils import open_mnist, process_img, write_h5


@click.command()
@click.argument('data_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
def main(data_path, out_path):
    # Load MNIST dataset
    path = data_path
    train_x, valid_x = open_mnist(path)

    # Process train images, split into x and y
    all_train = np.concatenate([process_img(img) for img in tqdm(train_x)])
    y_train = all_train[0::2]
    x_train = all_train[1::2]

    # Process valid images, split into x and y
    all_valid = np.concatenate([process_img(img) for img in tqdm(valid_x)])
    y_valid = all_valid[0::2]
    x_valid = all_valid[1::2]

    # Write data to h5 file
    outpath_train = str(out_path) + '/mnist_train.h5'
    write_h5(outpath_train, x_train, y_train, 'x_train', 'y_train')
    outpath_valid = str(out_path) + '/mnist_valid.h5'
    write_h5(outpath_valid, x_valid, y_valid, 'x_valid', 'y_valid')


if __name__ == '__main__':
    main()
