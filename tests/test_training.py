from click.testing import CliRunner


def test_create_databunch():
    from radionets.dl_framework.data import load_data, DataBunch, get_dls

    data_path = "./tests/build/data/"
    source_list = False
    fourier = False
    batch_size = 64

    # Load data sets
    train_ds = load_data(data_path, "train", source_list=source_list, fourier=fourier)
    valid_ds = load_data(data_path, "valid", source_list=source_list, fourier=fourier)

    # Create databunch with defined batchsize
    data = DataBunch(*get_dls(train_ds, valid_ds, batch_size))

    assert data.train_dl is not None
    assert data.valid_dl is not None
    assert data.c is None


def test_define_learner():
    from radionets.dl_framework.learner import define_learner
    import toml
    from radionets.dl_training.utils import read_config, define_arch, create_databunch

    config = toml.load("./tests/training.toml")
    train_conf = read_config(config)

    arch = define_arch(arch_name=train_conf["arch_name"], img_size=63)
    data = create_databunch(
        data_path=train_conf["data_path"],
        fourier=train_conf["fourier"],
        batch_size=train_conf["batch_size"],
        source_list=train_conf["source_list"],
    )

    learn = define_learner(data, arch, train_conf)

    assert learn.loss_func is not None
    assert str(learn.cbs[0]) == "TrainEvalCallback"
    assert str(learn.cbs[1]) == "Recorder"


def test_training():
    from radionets.dl_training.scripts.start_training import main

    runner = CliRunner()
    options = ["tests/training.toml"]
    result = runner.invoke(main, options)
    assert result.exit_code == 0


def test_save_model():
    import torch
    from collections.abc import Mapping

    def check(x):
        if torch.is_tensor(x):
            assert x.nelement != 0
            assert ~x.isnan().any()
            assert ~(0 in x)

        assert x is not None

    model = torch.load("tests/build/test_training/test_training.model")
    fastai_list = type(model["opt"]["hypers"])

    for key, value in model.items():
        if isinstance(value, Mapping):
            for key2, value2 in value.items():
                if torch.is_tensor(value2) or isinstance(value2, int):
                    check(value2)
                elif isinstance(value2, list) or type(value2) == fastai_list:
                    for ele3 in value2:
                        for key4, value4 in ele3.items():
                            if torch.is_tensor(value4) or isinstance(
                                value4, (int, float)
                            ):
                                check(value4)
                            else:
                                assert False, f"Unrecognised type {key4} {type(value4)}"
                else:
                    assert False, f"Unrecognised type {key2} {type(value2)}"
        elif isinstance(value, list):
            for ele2 in value:
                if isinstance(ele2, (int, float)) or torch.is_tensor(ele2):
                    check(ele2)
                elif type(ele2) == fastai_list:
                    for ele3 in ele2:
                        if isinstance(ele3, (float)):
                            check(ele3)
                        else:
                            assert False, f"Unrecognised type {type(ele3)}"
                else:
                    assert False, f"Unrecognised type {type(ele2)}"
        else:
            check(value)


def test_load_pretrained_model():
    import toml
    from radionets.dl_training.scripts.start_training import main

    config = toml.load("tests/training.toml")
    config["paths"]["pre_model"] = config["paths"]["model_path"]
    config["paths"]["model_path"] = config["paths"]["model_path"] + "_2"
    with open("tests/build/tmp_training.toml", "w") as toml_file:
        toml.dump(config, toml_file)

    runner = CliRunner()
    options = ["tests/build/tmp_training.toml"]
    result = runner.invoke(main, options)

    assert result.exit_code == 0


def test_plot_loss():
    from radionets.dl_training.scripts.start_training import main

    runner = CliRunner()
    options = ["tests/training.toml", "--mode=plot_loss"]
    result = runner.invoke(main, options)
    assert result.exit_code == 0
