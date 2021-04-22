from click.testing import CliRunner


def test_training():
    from radionets.dl_training.scripts.start_training import main
    runner = CliRunner()
    options = ["new_tests/training.toml"]
    result = runner.invoke(main, options)
    assert result.exit_code == 0


def test_plot_loss():
    from radionets.dl_training.scripts.start_training import main
    runner = CliRunner()
    options = ["new_tests/training.toml", "--mode=plot_loss"]
    result = runner.invoke(main, options)
    assert result.exit_code == 0
