import click
import warnings
import numpy as np
from mnist_cnn.utils import get_h5_data, mean_and_std
import pandas as pd


@click.command()
@click.argument('train_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
def main(train_path, out_path):
    x_train, _ = get_h5_data(train_path, columns=['x_valid', 'y_valid'])
    
    train_mean, train_std = mean_and_std(x_train)
    d = {'train_mean': train_mean, 'train_std': train_std}
    df = pd.DataFrame(data=d, index=[0])
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
