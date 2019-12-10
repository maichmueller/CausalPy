from functools import reduce

import numpy as np
import pandas as pd
from test.builld_scm_funcs import build_scm_medium
from causalpy.causal_prediction import GLMPredictor
from statsmodels.api import families


def to_count(sample, rs):
    sample = pd.DataFrame(
        rs.poisson(np.exp(sample)), columns=sample.columns
    )
    return sample


def test_icp_glm():
    cn = build_scm_medium(seed=0)

    rs = np.random.default_rng(8)

    cn_vars = list(cn.get_variables())
    data_unintervend = to_count(cn.sample(1000)[cn_vars], rs)
    cn.do_intervention(["X_3"], [0])
    data_intervention_1 = to_count(cn.sample(1000)[cn_vars], rs)
    cn.undo_intervention()
    cn.do_intervention(["X_0"], [0.23])
    data_intervention_2 = to_count(cn.sample(1000)[cn_vars], rs)
    cn.undo_intervention()
    cn.do_intervention(["X_1"], [-0.86])
    data_intervention_3 = to_count(cn.sample(1000)[cn_vars], rs)
    cn.undo_intervention()
    cn.do_intervention(["X_2"], [0.34])
    data_intervention_4 = to_count(cn.sample(1000)[cn_vars], rs)
    cn.undo_intervention()
    cn.do_intervention(["X_4"], [-.9])
    data_intervention_5 = to_count(cn.sample(1000)[cn_vars], rs)

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

    causal_parents, p_vals = GLMPredictor(
        glm_family=families.Poisson(), alpha=0.1, log_level="DEBUG"
    ).infer(obs, envs, target)

    assert sorted(causal_parents) == sorted(cn.graph.predecessors(target))