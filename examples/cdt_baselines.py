import networkx as nx
from cdt.causality.graph import GIES, PC
from cdt.data import load_dataset
from causalpy.datasets.cropseq_small import load_cropseq_small
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # data, graph = load_dataset("sachs")
    data, target, envs, _ = load_cropseq_small()
    obj = GIES()
    # The predict() method works without a graph, or with a
    # directed or undirected graph provided as an input
    output = obj.predict(data)  # No graph provided as an argument

    # To view the graph created, run the below commands:
    nx.draw_networkx(output, font_size=8)
    plt.show()

    obj = PC()
    # The predict() method works without a graph, or with a
    # directed or undirected graph provided as an input
    output = obj.predict(data)  # No graph provided as an argument

    # To view the graph created, run the below commands:
    nx.draw_networkx(output, font_size=8)
    plt.show()
