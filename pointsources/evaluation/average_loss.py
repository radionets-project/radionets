import click
import pandas as pd
import numpy as np
from mnist_cnn.visualize.utils import eval_model
from mnist_cnn.utils import get_h5_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dl_framework.architectures as architecture
from dl_framework.model import load_pre_model
from dl_framework.data import do_normalisation
from mnist_cnn.utils import split_real_imag, combine_and_swap_axes

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
        x_valid, y_valid = torch.Tensor(x_valid).view(-1,1,64,64), torch.Tensor(y_valid).view(-1,1,64,64)
        total = len(x_valid)
           
        #get arch
        arch = getattr(architecture, arch)()

        #load trained model
        load_pre_model(arch, pretrained_path, visualize=True)

        #norm x_valid
        norm = pd.read_csv(norm_path)
        img_normed = [do_normalisation(x_valid[i].view(1,1,64,64), norm) for i in range(total)]
        img_normed = torch.tensor(np.stack(img_normed))

        prediction = [eval_model(img_normed[i].view(1,1,64,64), arch) for i in range(total)]
        prediction = torch.tensor(np.stack(prediction))
        correct = 0
        false = 0
        loss = nn.MSELoss()
        prediction = eval_model(x_valid, arch)
        average_loss = loss(prediction, y_valid.view(-1,4096))


        for i in range(total):
             difference = loss(prediction[i], y_valid[i].view(4096))
             if difference <= average_loss:
                  correct += 1
             else:
                  false += 1

        print('Number of test imagese:', len(x_valid))
        print('Number of correct constructed images:', correct, correct/len(x_valid), "%")
        print('Number of false constructed images:', false, false/len(x_valid), "%")


        d = {'correct': [correct,correct/total], 'false': [false,false/total],'total' : [total,total/total]}
        df = pd.DataFrame(data=d)
        df.to_csv(out_path, index=False)

if __name__ == '__main__':
    main()

