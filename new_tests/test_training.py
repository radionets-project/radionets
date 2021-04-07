import subprocess


def test_lr_find():
    list_files = subprocess.run(
        ["radionets_training", "new_tests/training_test.toml", "--mode=lr_find"]
    )
    print("The exit code was: %d" % list_files.returncode)
    assert list_files.returncode == 0


def test_training():
    list_files = subprocess.run(["radionets_training", "new_tests/training_test.toml"])
    print("The exit code was: %d" % list_files.returncode)
    assert list_files.returncode == 0


def test_plot_loss():
    list_files = subprocess.run(
        ["radionets_training", "new_tests/training_test.toml", "--mode=plot_loss"]
    )
    print("The exit code was: %d" % list_files.returncode)
    assert list_files.returncode == 0
