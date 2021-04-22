from click.testing import CliRunner


def test_create_databunch():
    from radionets.dl_framework.data import load_data, DataBunch, get_dls

    data_path = "./new_tests/build/data/"
    source_list = False
    fourier = False
    batch_size = 64

    # Load data sets
    train_ds = load_data(data_path, "train", source_list=source_list, fourier=fourier)
    valid_ds = load_data(data_path, "valid", source_list=source_list, fourier=fourier)

    # Create databunch with defined batchsize
    bs = batch_size
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))

    assert data.train_dl is not None
    assert data.valid_dl is not None
    assert data.c is None


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
