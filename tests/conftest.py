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

@pytest.fixture(autouse=False)
def run_around_tests():
    with oce.Remote("http://api.oloren.ai:5000") as remote:
        yield
