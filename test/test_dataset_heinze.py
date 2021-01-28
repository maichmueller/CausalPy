from causalpy import SignSqrtAssignment
from causalpy.datasets import HeinzeData


def test_heinze():
    h = HeinzeData(seed=0)

    h.draw_config()
    data = h.sample()
    assert True

    config = dict(
        sample_size=500,
        target=f"X_{3}",
        noise_df=5,
        multiplicative=True,
        shift=True,
        meanshift=0.2,
        strength=0.5,
        mechanism=SignSqrtAssignment,
        interventions="all",
    )
