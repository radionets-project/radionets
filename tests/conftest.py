import shutil

import pytest


@pytest.fixture(autouse=True, scope="session")
def test_suite_cleanup_thing():
    yield

    build = "./tests/build/"
    print("Cleaning up tests.")

    shutil.rmtree(build)
