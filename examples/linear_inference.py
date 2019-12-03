from causalpy import LinICP
from simulation import *

if __name__ == '__main__':
    causal_net = simulate(5, 2, seed=0)
    print(causal_net)
    rs = np.random.default_rng(0)
    vars = list(causal_net.get_variables())
    n_normal = rs.integers(100, 5000+1)
    obs = [causal_net.sample(n_normal)[vars]]
    envs = [0] * n_normal
    curr_env = 1
    target_variable = rs.choice(causal_net.get_variables(False), size=1)[0]
    for variable in vars:
        if variable != target_variable:
            n_interv = rs.integers(100, 5000+1)

            causal_net.do_intervention([variable], [rs.integers(-100, 100)])

            obs.append(causal_net.sample(n_interv)[vars])
            envs += [curr_env] * n_interv

            causal_net.undo_intervention()
            curr_env += 1

    obs = pd.concat(obs, axis=0).reset_index(drop=True)
    envs = np.array(envs)
    linicp = LinICP(alpha=0.01, filter_method="lasso_sqrt", log_level="DEBUG")
    predicted_parents, p_vals = linicp.infer(obs, target_variable=target_variable, envs=envs)
    actual_parents = list(causal_net.graph.predecessors(target_variable))
    print("Target:", target_variable)
    print("Actual parents:", sorted(actual_parents))
    print("Predicted parents:", sorted(predicted_parents))
