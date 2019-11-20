import click
import torch
import torch.nn as nn
from mnist_cnn.visualize.utils import load_architecture
from functools import partial
import matplotlib.pyplot as plt
from preprocessing import get_h5_data, prepare_dataset, get_dls, DataBunch
from dl_framework.model import init_cnn
from dl_framework.param_scheduling import sched_no
from dl_framework.callbacks import Recorder, AvgStatsCallback,\
                                   BatchTransformXCallback, CudaCallback,\
                                   SaveCallback, view_tfm, ParamScheduler,\
                                   normalize_tfm
from inspection import evaluate_model
from dl_framework.learner import Learner
from dl_framework.optimizer import sgd_opt
from dl_framework.optimizer import (StatefulOptimizer, weight_decay,
                                    AverageGrad)
from dl_framework.optimizer import adam_step, AverageSqrGrad, StepCount


@click.command()
@click.argument('train_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('valid_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('arch_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('norm_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('num_epochs', type=int)
@click.argument('lr', type=float)
@click.option('-log', type=bool, required=False)
@click.option('-pretrained', type=bool, required=False)
@click.option('-inspection', type=bool, required=False)
@click.argument('pretrained_model',
                type=click.Path(exists=True, dir_okay=True), required=False)
def main(train_path, valid_path, model_path, arch_path, norm_path, num_epochs,
         lr, log=True, pretrained=False, pretrained_model=None,
         inspection=False):
    # Load data
    x_train, y_train = get_h5_data(train_path, columns=['x_train', 'y_train'])
    x_valid, y_valid = get_h5_data(valid_path, columns=['x_valid', 'y_valid'])

    # Create train and valid datasets
    train_ds, valid_ds = prepare_dataset(x_train, y_train, x_valid, y_valid,
                                         log=log)

    # Create databunch with defined batchsize
    bs = 256
    data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)

    # Define model
    get_model = load_architecture(arch_path)

    # Define resize for mnist data
    mnist_view = view_tfm(1, 64, 64)

    # make normalisation
    norm = normalize_tfm(norm_path)

    # Define scheduled learning rate
    sched = sched_no(lr, lr)

    # Define callback functions
    cbfs = [
        Recorder,
        partial(AvgStatsCallback, nn.MSELoss()),
        partial(ParamScheduler, 'lr', sched),
        CudaCallback,
        partial(BatchTransformXCallback, norm),
        partial(BatchTransformXCallback, mnist_view),
        SaveCallback,
    ]

    def get_learner(data, lr, loss_func=nn.MSELoss(),
                    cb_funcs=None, opt_func=sgd_opt, **kwargs):
        model = get_model
        init_cnn(model)
        return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs,
                       opt_func=opt_func)

    adam_opt = partial(StatefulOptimizer, steppers=[adam_step, weight_decay],
                       stats=[AverageGrad(dampening=True), AverageSqrGrad(),
                       StepCount()])

    # Combine model and data in learner
    learn = get_learner(data, 1e-3, opt_func=adam_opt,  cb_funcs=cbfs)

    if pretrained is True:
        # Load model
        name_pretrained = pretrained_model.split("/")[-1].split(".")[0]
        print('\nLoad pretrained model: {}\n'.format(name_pretrained))
        m = learn.model
        m.load_state_dict((torch.load(pretrained_model)))

    # Train model
    learn.fit(num_epochs)

    # Save model
    state = learn.model.state_dict()
    torch.save(state, model_path)

    if inspection is True:
        evaluate_model(valid_ds, learn.model, norm_path)
        plt.savefig('inspection_plot.pdf', dpi=300, bbox_inches='tight',
                    pad_inches=0.01)


if __name__ == '__main__':
    main()
