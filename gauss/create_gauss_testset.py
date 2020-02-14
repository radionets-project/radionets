import click
import numpy as np
from tqdm import tqdm
from utils import process_img, write_h5, pointsources, gauss, create_gauss
from sampling.uv_simulations import sample_freqs



@click.command()
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('num_img', type=int)
@click.argument('sources', type=int)
@click.argument('spherical', type=bool)
@click.argument('antenna_config_path', type=click.Path(exists=True,
                                                       dir_okay=True))
@click.option('-noise', type=bool, required=False)
@click.option('-specific_mask', type=bool)
@click.option('-lon', type=float, required=False)
@click.option('-lat', type=float, required=False)
@click.option('-steps', type=float, required=False)

def main(out_path, num_img, sources,spherical, antenna_config_path, noise=False, specific_mask=False,
         lon=None, lat=None, steps=None):
    
    # Create num_img pointsources images with targets sources
    test_x = create_gauss(num_img,sources,spherical)

    # Process test images
    all_test = np.concatenate([process_img(img, noise) for img in tqdm(test_x)])
    y_test = np.abs(all_test[0::2])
    x_test = all_test[1::2]

    if specific_mask is True:
        x_samp = [sample_freqs(i, antenna_config_path, lon, lat, steps)
                  for i in tqdm(x_test)]
    else:
        x_samp = [sample_freqs(i, antenna_config_path) for i in tqdm(x_test)]
        y_samp = y_test
        
    x_samp = np.array([abs(np.fft.ifft2(np.fft.ifftshift(x_samp[i].reshape(64,64)))) for i in range(len(x_samp))])
    x_samp = x_samp.reshape(-1,4096)
    
    write_h5(out_path, x_samp, y_samp, 'x_valid', 'y_valid')


if __name__ == '__main__':
    main()
