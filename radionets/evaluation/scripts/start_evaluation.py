import click
import toml
from radionets.evaluation.utils import read_config
from radionets.evaluation.train_inspection import (
    create_inspection_plots,
    create_source_plots,
)


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    """
    Start evaluation of trained deep learning model.

    Parameters
    ----------
    configuration_path: str
        Path to the configuration toml file
    """
    conf = toml.load(configuration_path)
    eval_conf = read_config(conf)

    click.echo("\n Evaluation config:")
    print(eval_conf, "\n")

    if eval_conf["vis_pred"]:
        create_inspection_plots(
            eval_conf, num_images=eval_conf["num_images"], rand=False
        )

        click.echo(f"\nCreated {eval_conf['num_images']} test predictions.\n")

    if eval_conf["vis_source"]:
        create_source_plots(eval_conf, num_images=eval_conf["num_images"], rand=False)

        click.echo(f"\nCreated {eval_conf['num_images']} source predictions.\n")
