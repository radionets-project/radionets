import click
from sampling.uv_simulations import sample_freqs
from mnist_cnn.utils import write_h5, get_h5_data
from tqdm import tqdm


@click.command()
@click.argument('data_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('antenna_config_path', type=click.Path(exists=True,
                                                       dir_okay=True))
@click.option('-train', type=bool)
def main(data_path, out_path, antenna_config_path, train=True):
    if train is True:
        x, y = get_h5_data(data_path, columns=['x_train', 'y_train'])
    else:
        x, y = get_h5_data(data_path, columns=['x_valid', 'y_valid'])

    x_samp = [sample_freqs(i, antenna_config_path) for i in tqdm(x)]
    y_samp = y

    if train is True:
        write_h5(out_path, x_samp, y_samp, 'x_train', 'y_train')
    else:
        write_h5(out_path, x_samp, y_samp, 'x_valid', 'y_valid')


if __name__ == '__main__':
    main()
