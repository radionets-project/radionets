import click
import pandas as pd
import numpy as np
from mnist_cnn.visualize.utils import eval_model
from mnist_cnn.utils import get_h5_data
import matplotlib.pyplot as plt
import torch
import dl_framework.architectures as architecture
from dl_framework.model import load_pre_model
from dl_framework.data import do_normalisation
from utils import num_blobs
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

@click.command()
@click.argument('arch', type=str)
@click.argument('pretrained_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('in_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('norm_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-threshold', type=float, required=False)
@click.option('-log', type=bool, required=False)
@click.option('-num', type=int, required=False)

def main(arch, pretrained_path, in_path, norm_path,
         out_path, threshold=None, log=False, num=None):

    x_valid, y_valid = get_h5_data(in_path, columns=['x_valid', 'y_valid'])
    x_valid, y_valid = torch.Tensor(x_valid).view(-1,1,64,64), torch.Tensor(y_valid).view(-1,1,64,64)

    total = len(x_valid)
    values = np.array([])
    correct = 0
    false = 0
     
    # get arch
    arch = getattr(architecture, arch)()

    # load pretrained model
    load_pre_model(arch, pretrained_path, visualize=True)

    # load the normalisation factors
    norm = pd.read_csv(norm_path)

    for i in tqdm(range(total)):
        img_reshaped = x_valid[i].view(1,1,64,64)
        img_normed = do_normalisation(img_reshaped, norm)
        #predict image
        prediction = eval_model(img_normed, arch)
        prediction = np.array(prediction.view(64,64))
        ground_truth = np.array(y_valid[i].view(64,64))

        #store detected blobs
        values = np.append(values, ssim(ground_truth, prediction))

    threshold = np.mean(values) - np.sqrt(np.var(values))
    
    for value in values:
        if value >= threshold:
            correct += 1
        else:
            false += 1

    print('threshold: {}'.format(threshold))
    print('Number of test imagese:', len(x_valid))
    print('Number of correct constructed images:', correct, correct/len(x_valid), "%")
    print('Number of false constructed images:', false, false/len(x_valid), "%")


    d = {'correct': [correct,correct/total], 'false': [false,false/total],'total' : [total,total/total]}
    df = pd.DataFrame(data=d)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
