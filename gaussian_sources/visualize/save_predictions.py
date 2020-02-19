import click
import re
import pandas as pd
import numpy as np
from dl_framework.data import get_bundles, h5_dataset
import dl_framework.architectures as architecture
from mnist_cnn.visualize.utils import eval_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from dl_framework.model import load_pre_model


@click.command()
@click.argument('data_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('arch', type=str)
@click.argument('pretrained_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-num', type=int, required=False)
@click.option('-fourier', type=bool, required=True)
def main(data_path, arch, pretrained_path, out_path, fourier, num=100):
    bundle_paths = get_bundles(data_path)
    test = [
        path for path in bundle_paths
        if re.findall('fft_samp_test', path.name)
        ]
    test_ds = h5_dataset(test, tar_fourier=fourier)
    indices = np.random.randint(0, len(test_ds), size=num)

    img_size = int(np.sqrt(test_ds[0][0].shape[1]))
    images = [test_ds[i][0].view(1, 2, img_size, img_size) for i in indices]

    arch = getattr(architecture, arch)()
    load_pre_model(arch, pretrained_path, visualize=True)

    prediction = [eval_model(img, arch).numpy().reshape(-1) for img in tqdm(images)]

    print(prediction[10].shape)

    outpath = str(out_path) + "predictions.csv"
    df = pd.DataFrame(prediction)
    df.to_csv(outpath, index=False)
    # plt.imshow(prediction[1].reshape(1, 2, 64, 64)[0, 1, :])
    # plt.show()


if __name__ == '__main__':
    main()
