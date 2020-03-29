import pytest
import os


@pytest.yield_fixture(autouse=True, scope='session')
def test_suite_cleanup_thing():
    yield

    build = "./tests/build/"
    print("Cleaning up tests.")

    if os.path.exists(build):
        for f in os.listdir(build):
            os.remove(build + f)
        os.rmdir(build)

    else:
        pass
