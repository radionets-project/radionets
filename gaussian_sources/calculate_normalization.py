import click
from dl_framework.data import (mean_and_std,
                               split_amp_phase,
                               )
from dl_framework.data import open_fft_pair, get_bundles
import pandas as pd
import numpy as np
import re
from tqdm import tqdm


@click.command()
@click.argument('data_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
def main(data_path, out_path):
    bundle_paths = get_bundles(data_path)
    bundle_paths = [path for path in bundle_paths
                    if re.findall('fft_samp_train', path.name)]

    means_real = np.array([])
    stds_real = np.array([])
    means_imag = np.array([])
    stds_imag = np.array([])

    for path in tqdm(bundle_paths):
        x, _ = open_fft_pair(path)
        # split in amp and phase
        x_real, x_imag = split_amp_phase(x)
        mean_real, std_real = mean_and_std(x_real)
        mean_imag, std_imag = mean_and_std(x_imag)
        means_real = np.append(mean_real, means_real)
        means_imag = np.append(mean_imag, means_imag)
        stds_real = np.append(std_real, stds_real)
        stds_imag = np.append(std_imag, stds_imag)

    mean_real = means_real.mean()
    std_real = stds_real.mean()
    mean_imag = means_imag.mean()
    std_imag = stds_imag.mean()

    d = {'train_mean_real': [mean_real],
         'train_std_real': [std_real],
         'train_mean_imag': [mean_imag],
         'train_std_imag': [std_imag]
         }

    df = pd.DataFrame(data=d)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
