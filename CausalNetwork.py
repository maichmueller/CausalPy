import numpy as np
import pandas as pd

import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

from typing import List, Union, Dict
from collections import deque
import matplotlib.pyplot as plt

class SCM:
    def __init__(self,
                 parents: List[List[int]],
                 functions: List[object],
                 noise_models: List[object],
                 variable_names: List[str] = None):

        self.roots = []
        self.nr_variables = len(parents)

        self.parents = parents
        self.functions = functions
        self.noise_models = noise_models
        if variable_names is not None:
            self.var_names = np.array(variable_names)
            self.var_names_draw = None
        else:
            self.var_names = np.asarray(["X_" + str(i) + "" for i in range(self.nr_variables)])
            self.var_names_draw = np.asarray([r"$X_{" + str(i) + r"}$" for i in range(self.nr_variables)])

        self.node_attributes = ["function", "noise", "data", "label"]

        self.graph = nx.DiGraph()

        for i, (name, parent_list, function, noise) in enumerate(
                zip(
                    self.var_names,
                    self.parents,
                    self.functions,
                    self.noise_models
                )
        ):
            self.graph.add_node(i, function=function, noise=noise, data=None, label=name)
            if parent_list:
                for parent in parent_list:
                    self.graph.add_edge(parent, i)
            else:
                self.roots.append(i)

    def __getitem__(self, item):
        return self.graph[item]

    def __str__(self):
        return self.print_causal_graph()

    def sample_graph(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)

        sample = pd.DataFrame([], index=range(n), columns=self.var_names)
        for node in self.traverse_from_roots():
            node_attr = self.graph.nodes[node]
            node_attr["data"] = node_attr["function"](
                node_attr["noise"](n),
                *(self.graph.nodes[pred]["data"] for pred in self.graph.predecessors(node))
            )
            sample.loc[:, node_attr['label']] = node_attr["data"]
        return sample

    def traverse_from_roots(self):
        next_nodes_queue = deque([self.roots])
        visited_nodes = []
        while next_nodes_queue:
            next_nodes = next_nodes_queue.popleft()
            for nn in next_nodes:
                next_nodes_queue.append(self.graph.successors(nn))
                if nn not in visited_nodes:
                    yield nn
                    visited_nodes.append(nn)

    def plot_causal_graph(self, node_size=1000, **kwargs):
        pos = graphviz_layout(self.graph, prog='dot')
        plt.title('Causal Network')
        if self.var_names_draw is not None:
            labels_dict = {label: draw_label for label, draw_label in zip(range(self.nr_variables), self.var_names_draw)}
            nx.draw(self.graph, pos, labels=labels_dict, with_labels=True, node_size=node_size, **kwargs)
        else:
            nx.draw(self.graph, pos, with_labels=True, node_size=1000, **kwargs)
        plt.show()

    def print_causal_graph(self):
        lines = [f"Structural Causal Model of {self.nr_variables} variables: " + ", ".join(self.var_names)]
        for node in range(self.nr_variables):
            lines.append(f"{self.var_names[node]} := {self.graph.nodes[node]['function']}")
        return "\n".join(lines)

