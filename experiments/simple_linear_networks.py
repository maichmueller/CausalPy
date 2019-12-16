from causalpy.causal_prediction.interventional import LinPredictor
from experiments.experiment_helpers import *
import numpy as np
import pandas as pd

if __name__ == "__main__":
    res = []
    n_iters = 100
    for scm_nr in range(100):
        try:
            causal_net = eval(f"build_simple_scm_{scm_nr}()")
        except NameError:
            break
        vars = list(causal_net.get_variables())

        iter_est_parents = []
        iter_est_pvals = []
        for i in range(n_iters):
            rs = np.random.default_rng()

            n_normal = rs.integers(100, 500 + 1)
            obs = [causal_net.sample(n_normal)[vars]]
            envs = [0] * n_normal

            target_variable = "Y"
            curr_env = 1
            for variable in vars:
                if variable != target_variable:
                    n_interv = rs.integers(100, 500 + 1)

                    causal_net.do_intervention(
                        [variable], [rs.random() * rs.choice([-1, 1]) * 10]
                    )

                    obs.append(causal_net.sample(n_interv)[vars])
                    envs += [curr_env] * n_interv

                    causal_net.undo_intervention()
                    curr_env += 1

            obs.append(causal_net.sample(n_interv)[vars])
            envs += [1] * n_interv

            causal_net.undo_intervention()

            obs = pd.concat(obs, axis=0).reset_index(drop=True)
            envs = np.array(envs)

            linicp = LinPredictor(
                alpha=0.05, filter_variables=False, log_level="INFO", residual_test="normal"
            )

            predicted_parents, p_vals = linicp.infer(
                obs, target_variable=target_variable, envs=envs
            )
            iter_est_parents.append(predicted_parents)
            iter_est_pvals.append(p_vals)

        actual_parents = list(causal_net.graph.predecessors(target_variable))

        print(f"Done estimating target Y from SCM {scm_nr}.")
        res.append(print_results(target_variable, iter_est_parents, actual_parents, as_dataframe=True))
    print(pd.concat(res).reset_index(drop=True))
