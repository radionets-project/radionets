import click
import re
import pandas as pd
import numpy as np
from dl_framework.data import get_bundles, h5_dataset
import dl_framework.architectures as architecture
from mnist_cnn.visualize.utils import eval_model
from tqdm import tqdm
import matplotlib.pyplot as plt


@click.command()
@click.argument('data_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('arch', type=str)
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-num', type=int, required=False)
def main(data_path, arch, out_path, num=100):
    print(data_path)
    bundle_paths = get_bundles(data_path)
    test = [
        path for path in bundle_paths
        if re.findall('fft_samp_test', path.name)
        ]
    test_ds = h5_dataset(test)
    indices = np.random.randint(0, len(test_ds), size=num)
    images = [test_ds[i][0].view(1, 2, 64, 64) for i in indices]
    arch = getattr(architecture, arch)()

    prediction = [eval_model(img, arch).view(4096).numpy() for img in tqdm(images)]

    print(prediction[1].shape)
    plt.imshow(prediction[1].reshape(64, 64))
    plt.show()
    # d = {'train_mean_real': [mean_real],
    #      'train_std_real': [std_real],
    #      'train_mean_imag': [mean_imag],
    #      'train_std_imag': [std_imag]
    #      }

    df = pd.DataFrame(prediction)
    df.to_csv('test.csv', index=False)


if __name__ == '__main__':
    main()
