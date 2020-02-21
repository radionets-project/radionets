import click
import pandas as pd
import numpy as np
from mnist_cnn.visualize.utils import eval_model
from mnist_cnn.utils import get_h5_data
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from skimage.feature import peak_local_max
import dl_framework.architectures as architecture
from dl_framework.model import load_pre_model
from dl_framework.data import do_normalisation

@click.command()
@click.argument('arch', type=str)
@click.argument('pretrained_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('in_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('norm_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-num', type=int, required=False)
def main(arch, pretrained_path, in_path, norm_path,
         out_path, num=None):
    
    x_valid, y_valid = get_h5_data(in_path, columns=['x_valid', 'y_valid'])
    x_valid, y_valid = torch.Tensor(x_valid).view(-1,1,64,64), torch.Tensor(y_valid).view(-1,1,64,64)
    
    total = len(x_valid)

    # get arch
    arch = getattr(architecture, arch)()

    # load pretrained model
    load_pre_model(arch, pretrained_path, visualize=True) 
    
    # load the normalisation factors
    norm = pd.read_csv(norm_path)
    
    indices = np.random.randint(0, total, size=num)
    img = x_valid[indices]
    
    print('\nPlotting {} pictures.\n'.format(num))

    for i in range(num):
        index = indices[i]
        img_reshaped = img[i].view(1,1,64,64)
        img_normed = do_normalisation(img_reshaped, norm)
        
        #predict image
        prediction = eval_model(img_normed, arch)
        
        #zero-padding
        prediction = np.pad(prediction.reshape(64,64),(2,2),'constant')
        ground_truth = np.pad(y_valid[index].reshape(64,64),(2,2),'constant')
        
        #plot
        coord_prediction = peak_local_max(prediction,threshold_abs=0.1)
        coord_ground_truth = peak_local_max(ground_truth, threshold_abs=0.1)

        fig, ((ax1,ax2)) = plt.subplots(1, 2, figsize=(20, 16))
        img1 = ax1.imshow(prediction)
        ax1.set_title('prediction')
        ax1.plot(coord_prediction[:,1], coord_prediction[:,0], marker='x', linestyle='none', color='red')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)    
        fig.colorbar(img1, cax=cax, orientation='vertical')

        img2 = ax2.imshow(ground_truth)
        ax2.set_title('ground truth')
        ax2.plot(coord_ground_truth[:,1], coord_ground_truth[:,0], marker='x', linestyle='none', color='red')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)    
        fig.colorbar(img2, cax=cax, orientation='vertical')

        outpath = str(out_path).split('.')[0] + '_{}.{}'.format(i, str(out_path).split('.')[-1]) 
        plt.savefig(str(outpath),bbox_inches='tight', pad_inches=0.01)
        plt.clf()
 

if __name__ == '__main__':
    main()       
