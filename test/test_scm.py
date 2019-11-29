from causalpy import LinearAssignment, PolynomialAssignment, NoiseGenerator, SCM
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial


def build_scm():
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 2),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 3, 2),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "X_3": (
                ["X_1", "X_2"],
                PolynomialAssignment([0, 1], [0, 1, 0.5], [0, 0, 4]),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "Y": (
                ["X_0", "X_2"],
                PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 2]),
                NoiseGenerator("standard_normal", seed=0),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
        },
    )
    return cn


def manual_standard_sample(n, noise_func, dtype, names):
    sample = np.empty((n, 5), dtype=dtype)
    sample[:, 0] = noise_func(n)
    sample[:, 1] = 1 + noise_func(n) + 2 * sample[:, 0]
    sample[:, 2] = 1 + noise_func(n) + 3 * sample[:, 0] + 2 * sample[:, 1]
    sample[:, 3] = (
        noise_func(n)
        + Polynomial([0, 1, 0.5])(sample[:, 1])
        + Polynomial([0, 0, 4])(sample[:, 2])
    )
    sample[:, 4] = (
        noise_func(n)
        + Polynomial([0, 0, 1.5])(sample[:, 0])
        + Polynomial([0, 2])(sample[:, 2])
    )
    sample = pd.DataFrame(sample, columns=names)
    return sample


def noise(n):
    return np.random.default_rng(0).standard_normal(n)


def test_linear_assignment():
    func = LinearAssignment(1, 0, 1, 2, 3)
    input_args = np.array([1, 2]), np.array([2, 4]), np.array([3, 6])

    assert (func(np.array([1, 1]), *input_args) == np.array([15, 29])).all()


def test_scm_build():
    cn = build_scm()
    nodes_in_graph = list(cn.graph.nodes)
    assert nodes_in_graph, ["X_0", "X_1", "X_2", "X_3", "Y"]


def test_scm_sample():
    cn = build_scm()
    scm_sample = cn.sample(10)
    sample = manual_standard_sample(
        10, noise, scm_sample.values.dtype, list(cn.graph.nodes)
    )
    sample_order_scm = list(cn.get_variables())
    sample = sample[sample_order_scm]
    # floating point inaccuracy needs to be accounted for
    assert (sample - scm_sample).abs().values.sum() < 1e-10


def test_scm_intervention():
    cn = build_scm()

    # do the intervention
    cn.intervention(
        {
            "X_3": {
                "parents": ["X_0", "Y"],
                "function": LinearAssignment(1, 0, 3.3, 3.3),
                "noise": NoiseGenerator("t", df=1, source="scipy", seed=0),
            }
        }
    )
    n = 10

    scm_sample_interv = cn.sample(n)
    sample = np.empty((n, 5), dtype=scm_sample_interv.values.dtype)
    sample[:, 0] = noise(n)
    sample[:, 1] = 1 + noise(n) + 2 * sample[:, 0]
    sample[:, 2] = 1 + noise(n) + 3 * sample[:, 0] + 2 * sample[:, 1]
    sample[:, 4] = (
        noise(n)
        + Polynomial([0, 0, 1.5])(sample[:, 0])
        + Polynomial([0, 2])(sample[:, 2])
    )
    sample[:, 3] = NoiseGenerator("t", df=1, source="scipy", seed=0)(n) + 3.3 * (
        sample[:, 0] + sample[:, 4]
    )
    sample = pd.DataFrame(sample, columns=list(cn.graph.nodes))
    sample_in_scm_order = sample[list(cn.get_variables())]

    assert (sample_in_scm_order - scm_sample_interv).abs().values.sum() < 1e-10

    # from here on the cn should work as normal again
    cn.undo_intervention()

    # reseeding needs to happen as the state of the initial noise distributions is further advanced than
    # a newly seeded noise by noise()
    cn.reseed(0)
    man_sample = manual_standard_sample(
        n, noise, scm_sample_interv.values.dtype, list(cn.graph.nodes)
    )[list(cn.get_variables())]
    new_cn_sample = cn.sample(n)
    assert (man_sample - new_cn_sample).abs().values.sum() < 1e-10
