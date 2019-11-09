import numpy as np
import pandas as pd
import logging
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

from typing import List, Union, Dict, Tuple
from collections import deque, defaultdict
import matplotlib.pyplot as plt


class SCM:
    def __init__(self,
                 assignment_dict: Dict[any, List[Tuple[List[any], object, object]]],
                 variable_tex_names: Dict = None):

        self.roots = []
        self.nr_variables = len(assignment_dict)

        self.var_names = np.array(list(assignment_dict.keys()))
        if variable_tex_names is not None:
            for name in self.var_names:
                if name not in variable_tex_names:
                    variable_tex_names[name] = name
        self.var_names_draw_dict = variable_tex_names

        self.var_name_to_node = dict()
        self.node_attributes = ["function", "noise"]

        self.graph = nx.DiGraph()
        for node_name, (parents_list, function, noise_model) in assignment_dict.items():
            self.graph.add_node(node_name, function=function, noise=noise_model)
            if parents_list:
                for parent in parents_list:
                    self.graph.add_edge(parent, node_name)
            else:
                self.roots.append(node_name)

    def __getitem__(self, item):
        return self.graph[item]

    def __str__(self):
        return self.str()

    def _causal_iterator(self, variables=None):
        if variables is None:
            return self.traverse_from_roots()
        visited_nodes = set()
        vars_to_sample_level = defaultdict(int)
        var_present = []
        for variable in variables:
            if variable not in self.var_names:
                logging.warning(f"Variable name {variable} unknown to naming list. Omitting it.")
            else:
                var_present.append(variable)
                vars_to_sample_level[variable] = 0
        queue = deque(var_present)
        while queue:
            nn = queue.popleft()
            if nn not in visited_nodes:
                for parent in self.graph.predecessors(nn):
                    vars_to_sample_level[parent] = max(vars_to_sample_level[parent], vars_to_sample_level[nn] + 1)
                    queue.append(parent)
                visited_nodes.add(nn)
        return (key for (key, value) in sorted(vars_to_sample_level.items(), key=lambda x: x[1], reverse=True))

    def sample(self, n, variables=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        sample = dict()
        for node in self._causal_iterator(variables):
            node_attr = self.graph.nodes[node]
            data = node_attr["function"](
                node_attr["noise"](n),
                *(sample[pred] for pred in self.graph.predecessors(node))
            )
            sample[node] = data
        return pd.DataFrame.from_dict(sample)

    def traverse_from_roots(self):
        next_nodes_queue = deque([self.roots])
        visited_nodes = set()
        while next_nodes_queue:
            next_nodes = next_nodes_queue.popleft()
            for nn in next_nodes:
                next_nodes_queue.append(self.graph.successors(nn))
                if nn not in visited_nodes:
                    yield nn
                    visited_nodes.add(nn)

    def plot(self, node_size=1000, **kwargs):
        pos = graphviz_layout(self.graph, prog='dot')
        plt.title('Causal Network')
        nx.draw(self.graph, pos, labels=self.var_names_draw_dict, with_labels=True, node_size=node_size, **kwargs)
        plt.show()

    def str(self):
        lines = [f"Structural Causal Model of {self.nr_variables} variables: " + ", ".join(self.var_names),
                 'Defined by the Structural Assignment Functions:']
        max_var_name_space = max([len(var_name) for var_name in self.var_names])
        for node in self.graph.nodes:
            parents_vars = [pred for pred in self.graph.predecessors(node)]
            line = f"{str(node).rjust(max_var_name_space)} := {self.graph.nodes[node]['function'].str(parents_vars)}"
            lines.append(line)
        lines.append("For a plot of the causal graph call 'plot' on the SCM object.")
        return "\n".join(lines)
