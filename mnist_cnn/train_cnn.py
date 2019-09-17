import click
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from preprocessing import get_h5_data, prepare_dataset, get_dls, DataBunch
from model import conv, init_cnn, Learner, Lambda, flatten
from training import view_tfm, sched_no, Recorder, AvgStatsCallback, ParamScheduler,\
                        BatchTransformXCallback, CudaCallback, Runner
from inspection import evaluate_model


from IPython import embed

@click.command()
@click.argument('train_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('valid_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('num_epochs', type=int)
@click.argument('lr', type=float)
@click.option('-log', type=bool, required=False)
@click.option('-quantile', type=bool, required=False)
@click.option('-pretrained', type=bool, required=False)
@click.option('-inspection', type=bool, required=False)
@click.argument('pretrained_model', type=click.Path(exists=True, dir_okay=True), required=False)
def main(train_path, valid_path, model_path, num_epochs, lr, log=True, quantile=False,
         pretrained=False, pretrained_model=None, inspection=False):
    # Load data
    x_train, y_train = get_h5_data(train_path, columns=['x_train', 'y_train'])
    x_valid, y_valid = get_h5_data(valid_path, columns=['x_valid', 'y_valid'])
    embed()
    # Create train and valid datasets
    train_ds, valid_ds = prepare_dataset(x_train, y_train, x_valid, y_valid, log=log, quantile=quantile)

    # # Create databunch with definde batchsize
    # bs = 128
    # data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)

    # # Define loss function
    # loss_func = nn.MSELoss()

    # # Define model
    # def get_model(data, lr=1e-1): #1e-1
    #     model = nn.Sequential(
    #         *conv(1, 4, (3,3), 1),
    #         *conv(4, 8, (3,3), 1),
    #         *conv(8, 16, (3,3), 1),
    #         nn.MaxPool2d((3,3)),
    #         *conv(16, 32, (2,2), 1),
    #         *conv(32, 64, (2,2), 1),
    #         nn.MaxPool2d((2,2)),
    #         Lambda(flatten),
    #         nn.Linear(4096, data.c)
    #     )
    #     return model, optim.SGD(model.parameters(), lr=lr)

    # # Combine model and data in learner
    # learn = Learner(*get_model(data), loss_func, data)

    # if pretrained is True:
    #     # Load model
    #     m = learn.model
    #     m.load_state_dict((torch.load(pretrained_model)))
    # else:
    #     # Initialize convolutional layers
    #     init_cnn(learn.model)

    # # Define resize for mnist data
    # mnist_view = view_tfm(1, 64, 64)

    # # Define schedueled learning rate
    # sched = sched_no(lr, lr)

    # # Define callback functions
    # cbfs = [
    #     Recorder,
    #     partial(AvgStatsCallback, loss_func),
    #     partial(ParamScheduler, 'lr', sched),
    #     CudaCallback,
    #     partial(BatchTransformXCallback, mnist_view),
    # ]

    # # Define runner
    # run = Runner(cb_funcs=cbfs)

    # # Train model
    # run.fit(num_epochs, learn)

    # # Save model
    # state = learn.model.state_dict()
    # torch.save(state, model_path)

    # if inspection is True:
    #     evaluate_model(valid_ds, learn.model)
    #     plt.savefig('inspection_plot.pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)

if __name__ == '__main__':
    main()