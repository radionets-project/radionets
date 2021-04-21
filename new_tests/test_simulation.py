# import subprocess
from click.testing import CliRunner


def test_simulation():
    from radionets.simulations.scripts.simulate_images import main

    runner = CliRunner()
    result = runner.invoke(main, "new_tests/simulate.toml")
    # list_files = subprocess.run(["radionets_simulations", "new_tests/simulate.toml"])
    # print("The exit code was: %d" % result)
    assert result.exit_code == 0
