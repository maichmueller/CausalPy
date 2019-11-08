from Assignments import LinearAssignment, PolynomialAssignment
from NoiseModels import NoiseGenerator
from CausalNetwork import SCM

if __name__ == '__main__':

    print(PolynomialAssignment([0, 1], [34, 5, .3], [3, 4, 6]))
    cn = SCM(
        parents=[[],
                 [0],
                 [0, 1],
                 [1, 2],
                 [0, 2]
                 ],
        functions=[LinearAssignment(1),
                   LinearAssignment(1, 1, 2),
                   LinearAssignment(1, 1, 3, 2),
                   PolynomialAssignment([0, 1], [0, 1, 1 / 2], [0, 0, 4]),
                   LinearAssignment(1, 1, 3 / 2, 2.)
                   ],
        noise_models=[NoiseGenerator("standard_normal")] * 5,
        variable_names=["$X_0$", "$X_1$", "$X_2$", "$X_3$", "Y"]

    )
    for node in cn.traverse_from_roots():
        print(node)
    cn.plot()
    sample = cn.sample_graph(10)
    print(sample)
    print(cn)
