import click
import warnings
import numpy as np
from mnist_cnn.utils import get_h5_data
from mnist_cnn.utils import create_mask
import pandas as pd


@click.command()
@click.argument('train_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('valid_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-log', type=bool, required=False)
@click.option('-use_mask', type=bool, required=False)
def main(train_path, valid_path, out_path, log=False, use_mask=False):
    x_train, _ = get_h5_data(train_path, columns=['x_train', 'y_train'])
    x_valid, _ = get_h5_data(valid_path, columns=['x_valid', 'y_valid'])

    if log is True:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        x_train = np.log(x_train)
        x_valid = np.log(x_valid)

    if use_mask is True:
        mask = create_mask(x_train)
        train_mean, _ = x_train[mask].mean(), x_train[mask].std()
    else:
        train_mean, _ = x_train.mean(), x_train.std()

    mask = create_mask(x_valid)
    valid_mean = x_valid[mask].mean()
    x_train[np.isinf(x_train)] = train_mean
    x_valid[np.isinf(x_valid)] = valid_mean
    train_std = x_train.std()

    d= {'mean': [train_mean], 'std': [train_std]}
    df = pd.DataFrame(data=d)
    df.to_csv(out_path)

if __name__ == '__main__':
    main()