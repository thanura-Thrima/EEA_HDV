from RTLib import core

def test_version():
    assert core.getVersion() == "0.0.0.1"