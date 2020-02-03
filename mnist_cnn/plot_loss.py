import click

import dl_framework.architectures as architecture
import torch
from dl_framework.callbacks import Recorder
from dl_framework.learner import get_learner
from dl_framework.model import load_pre_model
from inspection import plot_loss


@click.command()
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("lr", type=float)
@click.argument("loss_func", type=str)
@click.argument(
    "pretrained_model", type=click.Path(exists=True, dir_okay=True), required=False
)
def main(
    model_path, arch, lr, loss_func, pretrained_model=None,
):
    data = []
    # Define model
    arch = getattr(architecture, arch)()
    cbfs = [
        Recorder,
    ]
    learn = get_learner(
        data, arch, lr=lr, opt_func=torch.optim.Adam, cb_funcs=cbfs, loss_func=loss_func
    )

    load_pre_model(learn, pretrained_model)

    # Plot loss
    plot_loss(learn, model_path)


if __name__ == "__main__":
    main()
