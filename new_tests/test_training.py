from click.testing import CliRunner
from radionets.dl_training.scripts.start_training import main


# def test_lr_find():
#     runner = CliRunner()
#     options = ["new_tests/training.toml", "--mode=lr_find"]
#     result = runner.invoke(main, options)
#     assert result.exit_code == 0


def test_training():
    runner = CliRunner()
    options = ["new_tests/training.toml"]
    result = runner.invoke(main, options)
    assert result.exit_code == 0


def test_plot_loss():
    runner = CliRunner()
    options = ["new_tests/training.toml", "--mode=plot_loss"]
    result = runner.invoke(main, options)
    assert result.exit_code == 0
