from functools import reduce

from causalpy.causal_prediction.interventional import ANMPredictor
from causalpy.bayesian_graphs.scm import SCM, LinkerAssignment, LinearAssignment, PolynomialAssignment, NoiseGenerator
from causalpy.neural_networks import FCNet
import numpy as np
import pandas as pd


def build_scm_linandpoly(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 2),
                NoiseGenerator("standard_normal", seed=seed+1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 3, 2),
                NoiseGenerator("standard_normal", seed=seed+2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                PolynomialAssignment([0, 1], [0, 1, 0.5], [0, 0, .4]),
                NoiseGenerator("standard_normal", seed=seed+3),
            ),
            "Y": (
                ["X_0", "X_2"],
                PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 2, .7, .2, .03]),
                NoiseGenerator("standard_normal", seed=seed+4),
            ),
            "X_4": (
                ["Y"],
                PolynomialAssignment([2, 1], [0, 0, .000005]),
                NoiseGenerator("standard_normal", seed=seed+4),
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


if __name__ == '__main__':
    cn = build_scm_linandpoly(0)
    cn_vars = list(cn.get_variables())
    data_unintervend = cn.sample(1000)[cn_vars]
    cn.do_intervention(["X_3"], [-4])
    data_intervention_1 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_0"], [5])
    data_intervention_2 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_1"], [-5])
    data_intervention_3 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_2"], [3.89])
    data_intervention_4 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_4"], [5])
    data_intervention_5 = cn.sample(1000)[cn_vars]

    obs = pd.concat(
        [
            data_unintervend,
            data_intervention_1,
            data_intervention_2,
            data_intervention_3,
            data_intervention_4,
            data_intervention_5,
        ],
        axis=0,
    ).reset_index(drop=True)

    envs = np.array(
        reduce(
            lambda x, y: x + y,
            [
                [i] * len(data)
                for i, data in zip(
                    range(6),
                    (
                        data_unintervend,
                        data_intervention_1,
                        data_intervention_2,
                        data_intervention_3,
                        data_intervention_4,
                        data_intervention_5,
                    ),
                )
            ],
        )
    )
    target = "Y"
    network = FCNet(len(cn_vars) - 1, 128, 64, 32, 16, 8, 1)
    anm_pred = ANMPredictor(network)
    inference = anm_pred.infer(obs, target, envs)