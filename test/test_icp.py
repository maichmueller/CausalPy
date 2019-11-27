
from causality import LinearAssignment, PolynomialAssignment, NoiseGenerator, SCM, LinICP
import numpy as np
import pandas as pd
from scipy import stats
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
                LinearAssignment(1, 0, 0.3, 0.7),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "Y": (
                ["X_0", "X_2", "X_3"],
                LinearAssignment(1, 3, -1, -2, 5),
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


def test_residual_test():
    rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
    rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
    rvs3 = stats.norm.rvs(loc=5, scale=20, size=500)
    rvs4 = stats.norm.rvs(loc=5, scale=20, size=100)
    rvs5 = stats.norm.rvs(loc=8, scale=20, size=100)
    p_1_2 = stats.ttest_ind(rvs1, rvs2, equal_var=False)[1]
    p_1_3 = stats.ttest_ind(rvs1, rvs3, equal_var=False)[1]
    p_1_4 = stats.ttest_ind(rvs1, rvs4, equal_var=False)[1]
    p_1_5 = stats.ttest_ind(rvs1, rvs5, equal_var=False)[1]

    assert LinICP.test_equal_gaussians(rvs1, rvs2)[0] == p_1_2
    assert LinICP.test_equal_gaussians(rvs1, rvs3)[0] == p_1_3
    assert LinICP.test_equal_gaussians(rvs1, rvs4)[0] == p_1_4
    assert LinICP.test_equal_gaussians(rvs1, rvs5)[0] == p_1_5





def test_linear_icp():
    cn = build_scm()

    data_unintervend = cn.sample(100)
    cn.do_intervention(["X_1"], [2])
    data_intervention_1 = cn.sample(100)
    cn.soft_intervention(["X_2"], [NoiseGenerator("standard_t", df=1, seed=0)])
    data_intervention_2 = cn.sample(100)

    obs = pd.concat([data_unintervend, data_intervention_1, data_intervention_2], axis=0).reset_index(drop=True)
    envs = np.array([0] * 100 + [1] * 100 + [2] * 100)
    target = "X_3"

    causal_parents, p_vals = LinICP().infer(obs, envs, target, alpha=0.05, prefilter_variables=False)

    assert causal_parents == tuple(cn.graph.predecessors(target))

