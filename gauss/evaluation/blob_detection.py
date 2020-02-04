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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

@click.command()
@click.argument('arch', type=str)
@click.argument('pretrained_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('in_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('norm_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-index', type=int, required=False)
@click.option('-log', type=bool, required=False)
@click.option('-num', type=int, required=False)
def main(arch, pretrained_path, in_path, norm_path,
         out_path, index=None, log=False, num=None):
    
    x_valid, y_valid = get_h5_data(in_path, columns=['x_valid', 'y_valid'])
    x_valid, y_valid = torch.tensor(x_valid).view(-1,1,64,64), torch.tensor(y_valid).view(-1,1,64,64)
    
    total = len(x_valid)
    correct = 0
    false = 0

    # get arch
    arch = getattr(architecture, arch)()

    # load pretrained model
    load_pre_model(arch, pretrained_path) 
    
    # load the normalisation factors
    norm = pd.read_csv(norm_path)
    
    for i in tqdm(range(total)):
        img_reshaped = x_valid[i].view(1,1,64,64)
        img_normed = do_normalisation(img_reshaped, norm)
        
        #predict image
        prediction = eval_model(img_normed, arch)
        
        #compare detected blobs
        if num_blobs(prediction) == num_blobs(y_valid[i]):
            correct += 1
        else:
            false += 1
        
    print('Number of test imagese:', len(x_valid))
    print('Number of correct constructed images:', correct, correct/len(x_valid), "%")
    print('Number of false constructed images:', false, false/len(x_valid), "%")


    d = {'total': [len(x_valid)], 'correct': [correct], 'false': [false]}
    df = pd.DataFrame(data=d)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
