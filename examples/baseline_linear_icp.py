from causalpy.causal_prediction.interventional import LinPredictor

import numpy as np
from build_scm_funcs import *
from linear_regression_eval import *


if __name__ == "__main__":

    seed = 0
    np.random.seed(seed)

    ###################
    # Data Generation #
    ###################
    pref = "single"
    for i, (scm_generator, target_var, fname) in enumerate(
        [
            # (build_scm_minimal, "Y", f"{pref}_min"),
            # (build_scm_minimal2, "Y", f"{pref}_min2"),
            # (build_scm_basic, "Y", f"{pref}_basic"),
            # (build_scm_basic_discrete, "Y", f"{pref}_basic_disc"),
            # (build_scm_exponential, "Y", f"{pref}_exp"),
            (build_scm_medium, "Y", f"{pref}_medium"),
            # (build_scm_large, "Y", f"{pref}_large"),
            # (build_scm_massive, "Y", f"{pref}_massive"),
            # (partial(simulate, nr_genes=100), "G_12", f"{pref}_sim100"),
            # (partial(simulate, nr_genes=20), "G_16", f"{pref}_sim20"),
            # (partial(simulate, nr_genes=25), "G_21", f"{pref}_sim25"),
            # (partial(simulate, nr_genes=30), "G_29", f"{pref}_sim30"),
        ]
    ):
        (
            complete_data,
            environments,
            scm,
            possible_parents,
            target_parents,
        ) = generate_data_from_scm(
            scm=scm_generator(seed=seed),
            countify=False,
            intervention_style="markov",
            target_var=target_var,
            sample_size=1024,
            seed=seed,
        )
        target_parents_indices = np.array(
            [possible_parents.index(par) for par in target_parents]
        )

        ap = LinPredictor()
        parents, p_vals = ap.infer(
            complete_data, target_var, environments, normalize=True
        )

        print("\nEstimated parents:", parents)
        print("Associated P vals:")
        print(f"{(p_vals *100).round(1)}\n")

        # evaluate(
        #     complete_data,
        #     ap,
        #     environments,
        #     ground_truth_assignment=scm[target_var][1][scm.function_key],
        #     x_vars=target_parents,
        #     targ_var=target_var,
        # )
