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
from utils import evaluate, flux_values
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
    ground_truth = y_valid.reshape(-1,64,64).numpy()
    prediction = np.stack(prediction).reshape(-1,64,64)

    correct = 0
    almost_correct = 0
    more_sources = 0
    fewer_sources = 0
    false_sources = 0
    more_and_false = 0
    few_and_false = 0

    indices_correct = []
    indices_almost = []

    for i in range(total):
        result = evaluate(prediction[i], ground_truth[i])

        if result[0]==0 and result[1]==0:
            correct += 1
            indices_correct.append(i)
        elif result[0]!=0 and result[1]==0:
            if result[0] > 0:
                more_sources += 1
            else:
                fewer_sources += 1
        elif result[0]==0 and result[1]!=0:
            if result[1] == -1:
                almost_correct +=1
                indices_almost.append(i)
            else:
                false_sources += 1
        else:
            if result[0] > 0:
                more_and_false += 1
            else:
                few_and_false += 1


    print('Size of test dataset: {}'.format(total))
    print('correct: {}'.format(correct))
    print('almost_correct: {}'.format(almost_correct))
    print('more_sources: {}'.format(more_sources))
    print('fewer_sources: {}'.format(fewer_sources))
    print('false_sources: {}'.format(false_sources))
    print('more_and_false: {}'.format(more_and_false))
    print('few_and_false: {}'.format(few_and_false))
    
    d = {'correct': [correct,correct/total], 
         'almost_correct': [almost_correct,almost_correct/total],
         'more_sources': [more_sources,more_sources/total],
         'fewer_sources': [fewer_sources,fewer_sources/total],
         'false_sources': [false_sources,false_sources/total],
         'more_and_false': [more_and_false,more_and_false/total],
         'few_and_false': [few_and_false,few_and_false/total],
         'total' : [total,total/total]}
    df = pd.DataFrame(data=d)
    df.to_csv(out_path, index=False)
    
    print(prediction.shape, ground_truth.shape)
    flux_correct = flux_values(prediction, ground_truth, indices_correct)
    flux_almost = flux_values(prediction, ground_truth, indices_almost)
    
    plt.hist(flux_correct, bins=np.arange(0.625,1.225,0.05), edgecolor='k')
    plt.title('flux_prediction/plux_ground_truth')
    plt.savefig('results/hist_flux_correct',bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    plt.hist(flux_almost, bins=np.arange(0.425,1.255,0.05), edgecolor='k')
    plt.title('flux_prediction/plux_ground_truth')
    plt.savefig('results/hist_flux_almost',bbox_inches='tight', pad_inches=0.01)
    plt.clf()


if __name__ == '__main__':
    main()       
