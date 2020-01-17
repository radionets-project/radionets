import click
import warnings
import numpy as np
from mnist_cnn.utils import get_h5_data
from mnist_cnn.utils import create_mask, split_real_imag, mean_and_std
import pandas as pd


@click.command()
@click.argument('train_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('valid_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-log', type=bool, required=False)
@click.option('-use_mask', type=bool, required=False)
def main(train_path, valid_path, out_path, log=False, use_mask=False):
    x_train, _ = get_h5_data(train_path, columns=['x_train', 'y_train'])
    
    # log option disabled, improve or drop in future versions
    
    # x_valid, _ = get_h5_data(valid_path, columns=['x_valid', 'y_valid'])

    # split in real and imaginary part
    #x_train_real, x_train_imag = split_real_imag(x_train)
    # x_valid_real, x_valid_imag = split_real_imag(x_valid)
    # print(x_train_real.shape)
    # print(x_train_imag.shape)

    # if log is True:
    #     warnings.filterwarnings("ignore", category=RuntimeWarning)
    #     x_train = np.log(x_train)
    #     x_valid = np.log(x_valid)

    # if use_mask is True:
    #     mask = create_mask(x_train)
    #     train_mean, _ = x_train[mask].mean(), x_train[mask].std()
    # else:
    #     train_mean, _ = x_train.mean(), x_train.std()

    # mask = create_mask(x_valid)
    # valid_mean = x_valid[mask].mean()
    # x_train[np.isinf(x_train)] = train_mean
    # x_valid[np.isinf(x_valid)] = valid_mean
    # train_std = x_train.std()
    #train_mean_real, train_std_real = mean_and_std(x_train_real)
    #train_mean_imag, train_std_imag = mean_and_std(x_train_imag)
    #d = {'train_mean_real': [train_mean_real],
    #     'train_std_real': [train_std_real],
    #     'train_mean_imag': [train_mean_imag],
    #     'train_std_imag': [train_std_imag]
    #     }
    train_mean, train_std = mean_and_std(x_train)
    d = {'train_mean': train_mean, 'train_std': train_std}
    df = pd.DataFrame(data=d, index=[0])
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
