from pathlib import Path
from click.testing import CliRunner
import traceback


def test_simulate_bundles():
    from gaussian_sources.simulate_bundles import main

    num_bundles_train = "2"
    num_bundles_valid = "1"
    num_bundles_test = "1"
    bundle_size = "5"
    img_size = "63"
    out_path = "./tests/build/gaussian_sources"
    Path(out_path).mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    options = [
        num_bundles_train,
        num_bundles_valid,
        num_bundles_test,
        bundle_size,
        img_size,
        out_path,
        "-noise",
        True,
    ]
    result = runner.invoke(main, options)
    print(traceback.print_exception(*result.exc_info))

    assert result.exit_code == 0
