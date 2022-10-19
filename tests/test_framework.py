import olorenchemengine as oce

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"


def remote(func):
    def wrapper(*args, **kwargs):
        with oce.Remote("http://api.oloren.ai:5000") as remote:
            func(*args, **kwargs)

    return wrapper


def test_main():
    oce.test_oce()


def test_config():
    assert isinstance(oce.CONFIG, dict)
    oce.set_config_param("TESTING PARAM", "1")
    assert oce.CONFIG["TESTING PARAM"] == "1"
    oce.test_oce()
    oce.remove_config_param("TESTING PARAM")
    assert "TESTING PARAM" not in oce.CONFIG
