import click
import sys
import toml
from radionets.dl_training.utils import (
    read_config,
    check_outpath,
    create_databunch,
    define_arch,
    pop_interrupt,
    end_training,
)
from radionets.dl_framework.learner import define_learner
from radionets.dl_framework.model import load_pre_model
from radionets.dl_framework.inspection import (
    plot_lr_loss,
    plot_loss,
    plot_lr,
)
from radionets.evaluation.train_inspection import after_training_plots
from pathlib import Path


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(
        ["train", "lr_find", "plot_loss", "fine_tune"], case_sensitive=False
    ),
    default="train",
)
def main(configuration_path, mode):
    """
    Start DNN training with options specified in configuration file.

    Parameters
    ----------
    configuration_path: str
        Path to the configuration toml file

    Modes
    -----
    train: start training of deep learning model (default option)
    lr_find: execute learning rate finder
    plot_loss: plot losscurve of existing model
    """
    config = toml.load(configuration_path)
    train_conf = read_config(config)

    click.echo("\n Train config:")
    print(train_conf, "\n")

    # create databunch
    data = create_databunch(
        data_path=train_conf["data_path"],
        fourier=train_conf["fourier"],
        batch_size=train_conf["bs"],
        source_list=train_conf["source_list"],
    )

    # get image size
    train_conf["image_size"] = data.train_ds[0][0][0].shape[1]

    # define architecture
    arch = define_arch(
        arch_name=train_conf["arch_name"], img_size=train_conf["image_size"]
    )

    if mode == "train":
        # check out path and look for existing model files
        check_outpath(train_conf["model_path"], train_conf)

        click.echo("Start training of the model.\n")

        # define_learner
        learn = define_learner(data, arch, train_conf)

        # load pretrained model
        if train_conf["pre_model"] != "none":
            learn.create_opt()
            load_pre_model(learn, train_conf["pre_model"])

        # Train the model, except interrupt
        # train_conf["comet_ml"] = True
        try:
            if train_conf["comet_ml"]:
                learn.comet.experiment.log_parameters(train_conf)
                with learn.comet.experiment.train():
                    learn.fit(train_conf["num_epochs"])
            else:
                learn.fit(train_conf["num_epochs"])
        except KeyboardInterrupt:
            pop_interrupt(learn, train_conf)

        end_training(learn, train_conf)

        if train_conf["inspection"]:
            after_training_plots(train_conf, rand=True)

    if mode == "fine_tune":
        click.echo("Start fine tuning of the model.\n")

        # define_learner
        learn = define_learner(
            data,
            arch,
            train_conf,
        )

        # load pretrained model
        if train_conf["pre_model"] == "none":
            click.echo("Need a pre-trained modle for fine tuning!")
            return

        learn.create_opt()
        load_pre_model(learn, train_conf["pre_model"])

        # Train the model, except interrupt
        try:
            learn.fine_tune(train_conf["num_epochs"])
        except KeyboardInterrupt:
            pop_interrupt(learn, train_conf)

        end_training(learn, train_conf)
        if train_conf["inspection"]:
            after_training_plots(train_conf, rand=True)

    if mode == "lr_find":
        click.echo("Start lr_find.\n")

        # define_learner
        learn = define_learner(data, arch, train_conf, lr_find=True)

        # load pretrained model
        if train_conf["pre_model"] != "none":
            learn.create_opt()
            load_pre_model(learn, train_conf["pre_model"])

        learn.lr_find()

        # save loss plot
        plot_lr_loss(
            learn,
            train_conf["arch_name"],
            Path(train_conf["model_path"]).parent,
            skip_last=5,
            output_format=train_conf["format"],
        )

    if mode == "plot_loss":
        click.echo("Start plotting loss.\n")

        # define_learner
        learn = define_learner(data, arch, train_conf, plot_loss=True)
        # load pretrained model
        if Path(train_conf["model_path"]).exists:
            load_pre_model(learn, train_conf["model_path"], plot_loss=True)
        else:
            click.echo("Selected model does not exist.")
            click.echo("Exiting.\n")
            sys.exit()

        plot_lr(
            learn, Path(train_conf["model_path"]), output_format=train_conf["format"]
        )
        plot_loss(
            learn, Path(train_conf["model_path"]), output_format=train_conf["format"]
        )


if __name__ == "__main__":
    main()
