import click
import numpy as np


@click.command()
@click.argument('in_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-noise', type=bool, required=False)
def main(in_path, out_path):
    '''
    get list of bundles
    get len of all all bundles
    split bundles into train and valid (factor 0.2?)
    for every bundle
        calculate fft -> create fft pairs
        save to new h5 file
    tagg train and valid in filename
    '''

    all_train = np.concatenate([process_img(img) for img in tqdm(train_x)])
    y_train = np.abs(all_train[0::2])
    x_train = all_train[1::2]


if __name__ == '__main__':
    main()


img_fft = np.fft.fftshift(np.fft.fft2(img_rescaled))