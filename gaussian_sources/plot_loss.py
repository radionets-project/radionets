import click

import dl_framework.architectures as architecture
import torch
from dl_framework.callbacks import Recorder
from dl_framework.learner import get_learner
from dl_framework.model import load_pre_model
from dl_framework.inspection import plot_loss, plot_lr


@click.command()
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
def main(
    model_path, arch,
):
    data = []
    # Define model
    arch = getattr(architecture, arch)()
    cbfs = [
        Recorder,
    ]
    learn = get_learner(
        data, arch, 1e-3, opt_func=torch.optim.Adam, cb_funcs=cbfs
    )

    load_pre_model(learn, model_path)

    # Plot loss
    plot_lr(learn, model_path)
    plot_loss(learn, model_path)


if __name__ == "__main__":
    main()
