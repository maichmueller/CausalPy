from causalpy.datasets import HeinzeData


def test_heinze():
    h = HeinzeData()
    obs, target, envs = h.sample()
    assert True
