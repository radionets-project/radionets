import click
import pandas as pd
import numpy as np
from mnist_cnn.visualize.utils import eval_model
from mnist_cnn.utils import get_h5_data
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from skimage.feature import blob_log
from utils import num_blobs, msssim
from skimage.metrics import structural_similarity as ss
import dl_framework.architectures as architecture
from dl_framework.model import load_pre_model
from dl_framework.data import do_normalisation
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
        
        #plot
        blobs_prediction = blob_log(prediction.reshape(64,64),min_sigma=0.1, max_sigma=7, num_sigma=30, threshold=0.4, overlap=0.9)
        blobs_ground_truth = blob_log(y_valid[index].reshape(64,64),min_sigma=0.1, max_sigma=7, num_sigma=30, threshold=0.4, overlap=0.9)
        #compute radii for the circle
        blobs_prediction[:, 2] = blobs_prediction[:, 2]*np.sqrt(2)
        blobs_ground_truth[:, 2] = blobs_ground_truth[:, 2]*np.sqrt(2)

        fig, ((ax1,ax2)) = plt.subplots(1, 2, figsize=(20, 16))
        img1 = ax1.imshow(prediction.reshape(64,64))
        ax1.set_title('prediction')
        for blob in blobs_prediction:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax1.add_patch(c)
            ax1.set_axis_off()
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)    
        fig.colorbar(img1, cax=cax, orientation='vertical')
        at = AnchoredText('Number of blobs: {}'.format(num_blobs(prediction)),
                          prop=dict(size=15), frameon=True,
                          loc='upper left',
                          )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        
        ax1.add_artist(at)
        info = AnchoredText('ssim: {:0.4f} \nmsssim: {:0.4f}'.format(ss(np.array(prediction.reshape(64,64)), np.array(y_valid[index].view(64,64))),
                                                    msssim(torch.Tensor(prediction).view(1,1,64,64),y_valid[index].view(1,1,64,64), normalize=True)),
                                                    prop=dict(size=15), frameon=True, loc='lower left')        
        info.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
        ax1.add_artist(info)
        img2 = ax2.imshow(y_valid[index].reshape(64,64))
        ax2.set_title('ground truth')
        for blob in blobs_ground_truth:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax2.add_patch(c)
            ax2.set_axis_off()
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)    
        fig.colorbar(img2, cax=cax, orientation='vertical')
        at = AnchoredText('Number of blobs: {}'.format(num_blobs(y_valid[index])),
                          prop=dict(size=15), frameon=True,
                          loc='upper left',
                          )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax2.add_artist(at)
        
        
        outpath = str(out_path).split('.')[0] + '_{}.{}'.format(i, str(out_path).split('.')[-1]) 
        plt.savefig(str(outpath),bbox_inches='tight', pad_inches=0.01)
        plt.clf()
 

if __name__ == '__main__':
    main()       
