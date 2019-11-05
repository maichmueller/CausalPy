import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Union, Dict
from collections import deque


class CausalNetwork:
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

        self.variable_names = np.array(variable_names) if variable_names is not None else np.arange(self.nr_variables)

        self.node_attributes = ["function", "noise", "data", "label"]

        self.graph = nx.DiGraph()

        for (name, parent_list, function, noise) in zip(variable_names, parents, functions, noise_models):
            self.graph.add_node(name, function=function, noise=noise, data=None, label=name)
            if parent_list:
                for parent in parent_list:
                    self.graph.add_edge(parent, name)
            else:
                self.roots.append(self.graph[name])

    def __getitem__(self, item):
        return self.graph[item]

    def sample_graph(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)

        sample = pd.DataFrame([], index=range(n), columns=self.variable_names)
        for node in self.traverse_from_roots():
            node["data"] = node["function"](
                *(pred["data"]
                  for pred in self.graph.predecessors(node)),
                node["noise"](n)
            )
            sample.loc[:, node['label']] = node["data"]
        return sample

    def traverse_from_roots(self):
        next_nodes_queue = deque(self.roots)
        while next_nodes_queue:
            next_nodes = next_nodes_queue.popleft()
            for nn in next_nodes:
                next_nodes_queue.append(self.graph.successors(nn))
                yield nn

    def plot_causal_graph(self, **kwargs):
        if self.variable_names.dtype == np.int:
            labels_dict = {}
            for node in self.traverse_from_roots():
                labels_dict[node["label"]] = r"$X_{" + str(node["label"]) + r"}$"
            nx.draw(self.graph, labels=labels_dict, with_labels=True, **kwargs)
        else:
            nx.draw(self.graph, with_labels=True, **kwargs)
