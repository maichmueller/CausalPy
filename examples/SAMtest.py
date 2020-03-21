import cdt
from cdt.causality.graph.SAM import SAM
from causalpy.bayesian_graphs import LinearAssignment, NoiseGenerator, SCM
import torch
from build_scm_funcs import build_scm_medium


if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    scm = build_scm_medium(0)
    data = scm.sample(10000)
    sam = SAM(train_epochs=100, test_epochs=10, nh=10, dnh=10, nruns=10, njobs=1)
    graph = sam.create_graph_from_data(data)
    nx.draw(graph)
    plt.show()
