import click
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
import matplotlib.pyplot as plt
from preprocessing import get_h5_data, prepare_dataset, get_dls, DataBunch
from dl_framework.model import conv, Lambda, flatten
from dl_framework.param_scheduling import sched_no
from dl_framework.callbacks import Recorder, AvgStatsCallback,\
                                   BatchTransformXCallback, CudaCallback,\
                                   SaveCallback, view_tfm, ParamScheduler
# from dl_framework.learner import get_learner
from inspection import evaluate_model


@click.command()
@click.argument('train_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('valid_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('num_epochs', type=int)
@click.argument('lr', type=float)
@click.option('-log', type=bool, required=False)
@click.option('-mask', type=bool, required=False)
@click.option('-pretrained', type=bool, required=False)
@click.option('-inspection', type=bool, required=False)
@click.argument('pretrained_model',
                type=click.Path(exists=True, dir_okay=True), required=False)
def main(train_path, valid_path, model_path, num_epochs, lr, log=True,
         mask=False, pretrained=False, pretrained_model=None,
         inspection=False):
    # Load data
    x_train, y_train = get_h5_data(train_path, columns=['x_train', 'y_train'])
    x_valid, y_valid = get_h5_data(valid_path, columns=['x_valid', 'y_valid'])

    # Create train and valid datasets
    train_ds, valid_ds = prepare_dataset(x_train[0:8], y_train[0:8], x_valid[0:8], y_valid[0:8],
                                         log=log, use_mask=mask)

    # Create databunch with definde batchsize
    bs = 128
    data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)

    # Define loss function
    loss_func = nn.MSELoss()

    # Define model
    def get_model(data, lr=1e-1):
        model = nn.Sequential(
            *conv(1, 4, (3, 3), 1, 0),
            *conv(4, 8, (3, 3), 1, 0),
            *conv(8, 16, (3, 3), 1, 0),
            nn.MaxPool2d((3, 3)),
            *conv(16, 32, (2, 2), 1, 0),
            *conv(32, 64, (2, 2), 1, 0),
            nn.MaxPool2d((2, 2)),
            Lambda(flatten),
            nn.Linear(4096, data.c)
        )
        return model

    # Define resize for mnist data
    mnist_view = view_tfm(1, 64, 64)

    # Define schedueled learning rate
    sched = sched_no(lr, lr)

    # Define callback functions
    cbfs = [
        Recorder,
        partial(AvgStatsCallback, loss_func),
        partial(ParamScheduler, 'lr', sched),
        CudaCallback,
        partial(BatchTransformXCallback, mnist_view),
        SaveCallback,
    ]

    from dl_framework.learner import Learner
    def get_learner(data, lr, opt_func, loss_func=torch.nn.MSELoss(),
                    cb_funcs=None, **kwargs):
        model = get_model(data, **kwargs)
        # init_cnn(model)
        return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs,
                       opt_func=opt_func)

    from dl_framework.optimizer import sgd_opt
    # Combine model and data in learner
    learn = get_learner(data, 1e-3, opt_func=sgd_opt, loss_func=nn.MSELoss(), cb_funcs=cbfs)

    if pretrained is True:
        # Load model
        print('Load pretrained model.')
        m = learn.model
        m.load_state_dict((torch.load(pretrained_model)))

    # Train model
    learn.fit(num_epochs, learn)

    # Save model
    state = learn.model.state_dict()
    torch.save(state, model_path)

    if inspection is True:
        evaluate_model(valid_ds, learn.model)
        plt.savefig('inspection_plot.pdf', dpi=300, bbox_inches='tight',
                    pad_inches=0.01)


if __name__ == '__main__':
    main()
