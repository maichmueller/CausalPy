import networkx as nx
from cdt.causality.graph import GIES
from cdt.data import load_dataset


if __name__ == "__main__":

    data, graph = load_dataset("sachs")
    obj = GIES()
    # The predict() method works without a graph, or with a
    # directed or undirected graph provided as an input
    output = obj.predict(data)  # No graph provided as an argument

    output = obj.predict(data, nx.DiGraph(graph))  # With an undirected graph

    output = obj.predict(data, graph)  # With a directed graph

    # To view the graph created, run the below commands:
    nx.draw_networkx(output, font_size=8)
    plt.show()
