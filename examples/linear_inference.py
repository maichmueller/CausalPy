from examples.simulation import *
from causalpy import LINGAMPredictor


def to_count_data(sample):
    # sample = pd.DataFrame(
    #     rs.poisson(np.log(1 + np.exp(sample))), columns=sample.columns
    # )
    # return sample
    return sample


if __name__ == "__main__":
    causal_net = simulate(10, 2, seed=3)
    print(causal_net)
    rs = np.random.default_rng(8)
    vars = list(causal_net.get_variables())
    n_normal = rs.integers(100, 5000 + 1)
    obs = [to_count_data(causal_net.sample(n_normal))[vars]]
    envs = [0] * n_normal
    curr_env = 1
    # target_variable = rs.choice(causal_net.get_variables(False), size=1)[0]
    target_variable = "G_3"
    for variable in vars:
        if variable != target_variable:
            n_interv = rs.integers(100, 500 + 1)

            causal_net.do_intervention(
                [variable], [rs.random() * rs.choice([-1, 1]) * 10]
            )

            obs.append(to_count_data(causal_net.sample(n_interv))[vars])
            envs += [curr_env] * n_interv

            causal_net.undo_intervention()
            curr_env += 1

    obs = pd.concat(obs, axis=0).reset_index(drop=True)
    obs.to_csv("rtest.csv", index=False)
    envs = np.array(envs)
    pd.Series(envs).to_csv("rtest_envs.csv", index=False)
    print(target_variable)
    linicp = LINGAMPredictor(
        alpha=0.05, filter_variables=False, log_level="DEBUG", residual_test="ranks"
    )
    predicted_parents, p_vals = linicp.infer(
        obs, target_variable=target_variable, envs=envs
    )
    actual_parents = list(causal_net.graph.predecessors(target_variable))
    print("Target:", target_variable)
    print("Actual parents:", sorted(actual_parents))
    print("Predicted parents:", sorted(predicted_parents))
