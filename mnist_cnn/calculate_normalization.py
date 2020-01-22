import click
from mnist_cnn.utils import get_h5_data
from mnist_cnn.utils import split_real_imag, mean_and_std
import pandas as pd


@click.command()
@click.argument('train_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('valid_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-log', type=bool, required=False)
@click.option('-use_mask', type=bool, required=False)
def main(train_path, valid_path, out_path, log=False, use_mask=False):
    x_train, _ = get_h5_data(train_path, columns=['x_train', 'y_train'])

    # split in real and imaginary part
    x_train_real, x_train_imag = split_real_imag(x_train)

    train_mean_real, train_std_real = mean_and_std(x_train_real)
    train_mean_imag, train_std_imag = mean_and_std(x_train_imag)
    d = {'train_mean_real': [train_mean_real],
         'train_std_real': [train_std_real],
         'train_mean_imag': [train_mean_imag],
         'train_std_imag': [train_std_imag]
         }
    df = pd.DataFrame(data=d)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
