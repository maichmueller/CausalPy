from examples.simulation import *
from experiments.experiment_helpers import *
from causalpy import LiNGAMPredictor
from statsmodels.api import families


def to_count_data(sample):
    # sample = pd.DataFrame(
    #     rs.poisson(np.log(1 + np.exp(sample))), columns=sample.columns
    # )
    return sample


if __name__ == "__main__":

    causal_net = simulate(5, 2, seed=0)
    vars = list(causal_net.get_variables())

    n_iters = 500
    iter_est_parents = []
    iter_est_pvals = []
    for i in range(n_iters):
        rs = np.random.default_rng()

        n_normal = rs.integers(100, 5000 + 1)
        obs = [to_count_data(causal_net.sample(n_normal))[vars]]
        envs = [0] * n_normal

        curr_env = 1
        target_variable = rs.choice(list(set(causal_net.get_variables(False)) - {"G_0", "G_1"}), size=1)[0]
        for variable in vars:
            if variable != target_variable:
                n_interv = rs.integers(100, 5000 + 1)

                causal_net.do_intervention(
                    [variable], [rs.random() * rs.choice([-1, 1]) * 10]
                )

                obs.append(to_count_data(causal_net.sample(n_interv))[vars])
                envs += [curr_env] * n_interv

                causal_net.undo_intervention()
                curr_env += 1

        obs = pd.concat(obs, axis=0).reset_index(drop=True)
        envs = np.array(envs)

        linicp = LiNGAMPredictor(
            alpha=0.05, filter_variables=False, log_level="INFO", residual_test="normal"
        )

        predicted_parents, p_vals = linicp.infer(
            obs, target_variable=target_variable, envs=envs
        )
        iter_est_parents.append(predicted_parents)
        iter_est_pvals.append(p_vals)

    actual_parents = list(causal_net.graph.predecessors(target_variable))
    print(causal_net)
    # obs.to_csv("rtest.csv", index=False)
    # pd.Series(envs).to_csv("rtest_envs.csv", index=False)
    print_results(target_variable, iter_est_parents, actual_parents, as_dataframe=False)
