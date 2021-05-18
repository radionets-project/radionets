import pytest
from click.testing import CliRunner


@pytest.mark.first
def test_simulation():
    from radionets.simulations.scripts.simulate_images import main

    runner = CliRunner()
    result = runner.invoke(main, "new_tests/simulate.toml")
    assert result.exit_code == 0
