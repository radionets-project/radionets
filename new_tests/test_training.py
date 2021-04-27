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


def test_define_learner():
    from radionets.dl_framework.learner import define_learner
    import toml
    from radionets.dl_training.utils import read_config, define_arch, create_databunch

    config = toml.load("./new_tests/training.toml")
    train_conf = read_config(config)

    arch = define_arch(arch_name=train_conf["arch_name"], img_size=63)
    data = create_databunch(
        data_path=train_conf["data_path"],
        fourier=train_conf["fourier"],
        batch_size=train_conf["bs"],
        source_list=train_conf["source_list"],
    )

    learn = define_learner(data, arch, train_conf)

    assert learn.loss_func is not None
    assert str(learn.cbs[0]) == "TrainEvalCallback"
    assert str(learn.cbs[1]) == "Recorder"


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
