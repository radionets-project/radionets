import subprocess


def test_simulation():
    list_files = subprocess.run(["radionets_simulations", "new_tests/simulate.toml"])
    print("The exit code was: %d" % list_files.returncode)
    assert list_files.returncode == 0
