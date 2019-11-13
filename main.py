from scm.assignments import LinearAssignment, PolynomialAssignment
from scm.noise_models import NoiseGenerator
from scm.scm import SCM

if __name__ == '__main__':

    cn = SCM(
        assignment_map={"X_0": ([],
                                LinearAssignment(1),
                                NoiseGenerator("negative_binomial", n=30, p=0.5)),
                         "X_1": (["X_0"],
                                 LinearAssignment(1, 1, 2),
                                 NoiseGenerator("negative_binomial", n=30, p=0.5)),
                         "X_2": (["X_0", "X_1"],
                                 LinearAssignment(1, 1, 3, 2),
                                 NoiseGenerator("negative_binomial", n=30, p=0.5)),
                        "X_3": (["X_1", "X_2"],
                                 PolynomialAssignment([0, 1], [0, 1, 1 / 2], [0, 0, 4]),
                                 NoiseGenerator("negative_binomial", n=30, p=0.5)),
                        "Y": (["X_0", "X_2"],
                               PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 1]),
                               NoiseGenerator("negative_binomial", n=30, p=0.5))},
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_2": "$X_2$", "X_3": "$X_3$"}
    )
    cn.plot()
    sample = cn.sample(100000)
    print(sample)
    print(sample.mean(axis=0))
    print(cn)
