import click
import pandas as pd
import numpy as np
from mnist_cnn.visualize.utils import eval_model
from mnist_cnn.utils import get_h5_data
import matplotlib.pyplot as plt
import torch
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
    rand = np.random.randint(len(x_valid))
    if index is None:
        indices = np.random.randint(0, len(x_valid), size=num)
        img = torch.tensor(x_valid[indices])
    else:
        img = torch.tensor(x_valid[index])

    if log is True:
        img = torch.log(img)

    # get arch
    arch = getattr(architecture, arch)()

    # load pretrained model
    load_pre_model(arch, pretrained_path) 
    
    if index is None:
        print('\nPlotting {} pictures.\n'.format(num))
        for i in range(num):
            index = indices[i]
            img_reshaped = img[i].view(1,1,64,64)
            norm = pd.read_csv(norm_path)
            img_normed = do_normalisation(img_reshaped, norm)
           
            
           
            
            #predict image
            prediction = eval_model(img_normed, arch)
            
            fig, ((ax1,ax2,ax3)) = plt.subplots(1,3,figsize=(20,16))
            inp = img_normed.numpy().reshape(64,64)
            im1 = ax1.imshow(inp)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax1.set_title(r'Input')
            fig.colorbar(im1, cax=cax, orientation='vertical')
            
            pred_img = prediction.reshape(64,64).numpy()
            im2 = ax2.imshow(pred_img)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax2.set_title(r'Prediction')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            im3 = ax3.imshow(y_valid[index].reshape(64,64))
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax3.set_title(r'Truth')
            fig.colorbar(im3, cax=cax, orientation='vertical')

            outpath = str(out_path).split('.')[0] + '_{}.{}'.format(i, str(out_path).split('.')[-1]) 
            plt.savefig(str(outpath),bbox_inches='tight', pad_inches=0.01)
            plt.clf()
            
    else:
        print('\nPlotting a single index.\n')
        img_reshaped = img.view(1,1,64,64)
        norm = pd.read_csv(norm_path)
        img_normed = do_normalisation(img_reshaped, norm)




        #predict image
        prediction = eval_model(img_normed, arch)

        fig, ((ax1,ax2,ax3)) = plt.subplots(1,3,figsize=(20,16))
        inp = img_normed.numpy().reshape(64,64)
        im1 = ax1.imshow(inp)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax3.set_title(r'Input')
        fig.colorbar(im1, cax=cax, orientation='vertical')

        pred_img = prediction.reshape(64,64).numpy()
        im2 = ax2.imshow(pred_img)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax2.set_title(r'Prediction')
        fig.colorbar(im2, cax=cax, orientation='vertical')

        im3 = ax3.imshow(y_valid[index].reshape(64,64))
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax3.set_title(r'Truth')
        fig.colorbar(im3, cax=cax, orientation='vertical')

        plt.savefig(str(out_path),bbox_inches='tight', pad_inches=0.01)
        plt.clf()     
        
        
if __name__ == '__main__':
    main()
