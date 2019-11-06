from Assignments import LinearFunc
from NoiseModels import NoiseGenerator
from CausalNetwork import SCM

if __name__ == '__main__':
    cn = SCM(
        parents=[[],
                 [0],
                 [0, 1],
                 [1, 2],
                 [0, 2]
                 ],
        functions=[LinearFunc(1),
                   LinearFunc(1, 1, 2),
                   LinearFunc(1, 1, 3, 2),
                   LinearFunc(1, 0, 1, 1/2),
                   LinearFunc(1, 1, 3/2, 0, 2.)
                   ],
        noise_models=[NoiseGenerator("standard_normal")] * 5

    )
    for node in cn.traverse_from_roots():
        print(node)
    cn.plot_causal_graph()
    sample = cn.sample_graph(10)
    print(sample)
    print(LinearFunc(1, 1, 3/2, 0, 2.))
    print(cn)