import click
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dl_framework.data import mean_and_std, get_bundles, open_fft_pair


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
def main(data_path, out_path):
    """
    Calculates normalization factors from the trainings data set.
    Saves normalizytion factors to csv.

    Parameters
    ----------
    data_path: str
        path to data directory
    out_path: str
        path to save norm factors in csv
    """
    bundle_paths = get_bundles(data_path)
    bundle_paths = [
        path for path in bundle_paths if re.findall("fft_bundle_samp_train", path.name)
    ]

    means_c0 = np.array([])
    stds_c0 = np.array([])
    means_c1 = np.array([])
    stds_c1 = np.array([])

    for path in tqdm(bundle_paths):
        x, _ = open_fft_pair(path)
        # split in channel 0 and cahnnel 1
        x_c0, x_c1 = x[:, 0], x[:, 1]
        mean_c0, std_c0 = mean_and_std(x_c0)
        mean_c1, std_c1 = mean_and_std(x_c1)
        means_c0 = np.append(mean_c0, means_c0)
        means_c1 = np.append(mean_c1, means_c1)
        stds_c0 = np.append(std_c0, stds_c0)
        stds_c1 = np.append(std_c1, stds_c1)

    mean_c0 = means_c0.mean()
    std_c0 = stds_c0.mean()
    mean_c1 = means_c1.mean()
    std_c1 = stds_c1.mean()

    d = {
        "train_mean_c0": [mean_c0],
        "train_std_c0": [std_c0],
        "train_mean_c1": [mean_c1],
        "train_std_c1": [std_c1],
    }

    df = pd.DataFrame(data=d)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
