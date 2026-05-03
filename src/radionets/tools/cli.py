import rich_click as click

from radionets import __version__

from .evaluation_cli import main as eval
from .model_cli import main as model
from .quickstart import main as quickstart

click.rich_click.COMMAND_GROUPS = {
    "radionets": [
        {
            "name": "Model Operations",
            "commands": ["train", "test", "inference", "predict"],
        },
        {
            "name": "Setup",
            "commands": ["quickstart"],
        },
    ]
}


@click.group(
    help=f"""
    This is the [dark_orange]Radionets[/]
    [cornflower_blue]v{__version__}[/] main CLI tool.
    """
)
def main():
    pass


def create_mode_command(mode, cmd_alias=None):
    """Factory function to create mode-specific commands"""
    if cmd_alias is None:
        cmd_alias = mode

    @click.command(name=cmd_alias, help=f"Run radionets in {mode} mode")
    @click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
    @click.pass_context
    def command(ctx, config_path):
        if mode == "train":
            ctx.invoke(model, config_path=config_path)
        elif mode in {"eval", "test"}:
            ctx.invoke(eval, config_path=config_path)

    return command


main.add_command(quickstart, name="quickstart")
main.add_command(create_mode_command("train"))
main.add_command(create_mode_command("test"))
main.add_command(create_mode_command("test", cmd_alias="eval"))
main.add_command(create_mode_command("predict"))
main.add_command(create_mode_command("predict", "inference"))  # NOTE: Subject to change

if __name__ == "__main__":
    main()
