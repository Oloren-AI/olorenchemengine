"""
    Dummy conftest.py for olorenchemengine.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

# import pytest

import pytest

import olorenchemengine as oce

# def pytest_configure(config):
#     config.addinivalue_line(
#         "markers",
#         "no_profiling: mark test to not use sql profiling"
#     )

# @pytest.fixture(scope="session", autouse=True)
# def cleanup(request):
#     """Cleanup a testing directory once we are finished."""
#     remote = oce.Remote("http://api.oloren.ai:5000")
#     print("Opening remote")
#     remote.__enter__()
#     def close_remote():
#         print("Closing remote")
#         remote.__exit__(None, None, None)
#     request.addfinalizer(close_remote)


@pytest.fixture(autouse=True)
def run_around_tests():
    with oce.Remote("http://api.oloren.ai:5000") as remote:
        yield
