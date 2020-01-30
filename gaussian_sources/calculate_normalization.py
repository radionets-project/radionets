import click
from gaussian_sources.preprocessing import split_amp_phase, mean_and_std, split_real_imag
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

    means_amp = np.array([])
    stds_amp = np.array([])
    means_phase = np.array([])
    stds_phase = np.array([])

    for path in tqdm(bundle_paths):
        x, _ = open_fft_pair(path)
        # split in amp and phase
        x_amp, x_phase = split_real_imag(x)
        mean_amp, std_amp = mean_and_std(x_amp)
        mean_phase, std_phase = mean_and_std(x_phase)
        means_amp = np.append(mean_amp, means_amp)
        means_phase = np.append(mean_phase, means_phase)
        stds_amp = np.append(std_amp, stds_amp)
        stds_phase = np.append(std_phase, stds_phase)

    mean_amp = means_amp.mean()
    std_amp = stds_amp.mean()
    mean_phase = means_phase.mean()
    std_phase = stds_phase.mean()

    d = {'train_mean_amp': [mean_amp],
         'train_std_amp': [std_amp],
         'train_mean_phase': [mean_phase],
         'train_std_phase': [std_phase]
         }

    df = pd.DataFrame(data=d)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
