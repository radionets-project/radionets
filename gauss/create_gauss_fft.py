import click
import numpy as np
from tqdm import tqdm
from utils import process_img, write_h5, pointsources, gauss, create_gauss


@click.command()
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('num_img', type=int)
@click.argument('sources', type=int)
@click.argument('spherical', type=bool)
@click.option('-test', type=bool, required=False)
@click.option('-noise', type=bool, required=False)

def main(out_path, num_img, sources,spherical, test=False, noise=False):
    
    if test:
        num_img = 50
    # Create num_img pointsources images with targets sources
    train_x, valid_x = create_gauss(num_img,sources,spherical), create_gauss(num_img//5, sources,spherical)

    # Check if its a test call
    # take only the first 50 pictures for a faster run

    # Process train images, split into x and y
    all_train = np.concatenate([process_img(img, noise) for img in tqdm(train_x)])
    y_train = np.abs(all_train[0::2])
    x_train = all_train[1::2]

    # Process valid images, split into x and y
    all_valid = np.concatenate([process_img(img, noise) for img in tqdm(valid_x)])
    y_valid = np.abs(all_valid[0::2])
    x_valid = all_valid[1::2]

    # Write data to h5 file
    outpath_train = str(out_path) + '/gauss_train.h5'
    write_h5(outpath_train, x_train, y_train, 'x_train', 'y_train')
    outpath_valid = str(out_path) + '/gauss_valid.h5'
    write_h5(outpath_valid, x_valid, y_valid, 'x_valid', 'y_valid')


if __name__ == '__main__':
    main()
