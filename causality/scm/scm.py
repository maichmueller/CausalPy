import numpy as np
import pandas as pd
import logging
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from . import Assignment, NoiseModel, NoiseGenerator, LinearAssignment, BaseAssignment

from typing import (
    List,
    Union,
    Dict,
    Tuple,
    Iterable,
    Set,
    Type,
    Mapping,
    Collection,
    TypeVar)
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from copy import deepcopy


class SCM:

    Assignment = TypeVar('Assignment', bound=BaseAssignment)
    NoiseModel = TypeVar('NoiseModel', bound=NoiseGenerator)

    def __init__(
        self,
        assignment_map: Mapping[
            object, Tuple[Collection, Type[Assignment], Type[NoiseModel]]
        ],
        variable_tex_names: Dict = None,
        function_key: str = "function",
        noise_key: str = "noise",
        scm_name: str = "Structural Causal Model",
    ):

        self.scm_name: str = scm_name
        # the root variables which are causally happening at first.
        self.roots: List = []
        self.nr_variables: int = len(assignment_map)

        self.var_names: np.ndarray = np.array(list(assignment_map.keys()))
        # supply any variable name, that has not been assigned a different TeX name, with itself as TeX name.
        # This prevents missing labels in the plot method.
        if variable_tex_names is not None:
            for name in self.var_names:
                if name not in variable_tex_names:
                    variable_tex_names[name] = name
            # the variable names as they can be used by the plot function to draw the names in TeX mode.
            self.var_names_draw_dict: Dict = variable_tex_names
        else:
            self.var_names_draw_dict = dict()

        # the attribute list that any given node in the graph has.
        self.function_key, self.noise_key = function_key, noise_key

        # a backup dictionary of the original assignments of the intervened variables,
        # in order to undo the interventions later.
        self.interventions_backup_attr: Dict = dict()
        self.interventions_backup_parent: Dict = dict()

        # build the graph:
        # any node will be given the attributes of function and noise to later sample from and also an incoming edge
        # from its causal parent to itself. We will store the causal root nodes separately.
        self.graph = nx.DiGraph()
        self._build_graph(assignment_map)

    def __getitem__(self, node):
        return self.graph.pred[node], self.graph.nodes[node]

    def __str__(self):
        return self.str()

    def sample(self, n, variables=None, seed=None):
        """
        Sample method to generate data for the given variables. If no list of variables is supplied, the method will
        simply generate data for all variables.
        Setting the seed guarantees reproducibility, however this also currently implies that all randomness is
        generated by a numpy method!

        :param n: int, number of samples
        :param variables: list, the variable names to consider for sampling. If None, all variables will be sampled.
        :param seed: int, the seeding for the numpy generator
        :return: pd.DataFrame, the dataframe containing the samples of all the variables needed for the selection.
        """
        if seed is not None:
            np.random.seed(seed)
        sample = dict()

        for node in self._causal_iterator(variables):
            node_attr = self.graph.nodes[node]
            data = node_attr[self.function_key](
                node_attr[self.noise_key](n),
                *(sample[pred] for pred in self.graph.predecessors(node)),
            )
            sample[node] = data
        np.random.seed(None)  # reset random seed
        return pd.DataFrame.from_dict(sample)

    def intervention(
        self, interventions: Dict[object, Union[Dict, List, Tuple, np.ndarray]]
    ):
        """
        Method to apply interventions on the specified variables.

        One can set any variable to a new assignment function and noise model, thus redefining its parents and their
        dependency structure. Using this method will enable the user to call sample, plot etc. on the SCM just like
        before.
        In order to allow for undoing the intervention(s), the original state of the variables in the network is saved
        as backup and can be undone by calling ``undo_interventions``.

        Parameters
        ----------
        interventions: dict,
            the variables as keys and their new assignment as values. For the values one can
            choose between a dictionary or a list-like.
            - For dict: the dictionary is assumed to have the optional keys (default behaviour explained):
                -- "parents": List of parent variables. If not given, set to current parents.
                -- "function_key": assignment functor. If not given, keeps current assignment function.
                -- "noise_key": noise model. If not given, keeps current noise model.

                Note, that when providing new parents without a new assignment function, the user implicitly assumes
                the order of positional parameters of the assignment function to agree with the iterative order of the
                new parents!

            - For list: the order is [Parent list, assignment functor, noise models]. In order to omit one of these, set
                them to None.
            - For tuple: same as list
            - For ndarray: same as list, but dim == 1 assumed (not checked).
        """
        for var, items in interventions.items():
            if var not in self.graph.nodes:
                logging.warning(f"Variable '{var}' not found in graph. Omitting it.")
                continue

            if isinstance(items, dict):
                if any(
                        (
                                key not in ("parents", self.function_key, self.noise_key)
                                for key in items.keys()
                        )
                ):
                    raise ValueError(
                        f"Intervention dictionary provided with the wrong keys.\n"
                        f"Observed keys are: {list(items.keys())}\n"
                        f"Possible keys are: ['parents', '{self.function_key}', '{self.noise_key}']"
                    )
                try:
                    parent_list = [
                        par for par in self._filter_variable_names(items.pop("parents"))
                    ]
                except KeyError:
                    parent_list = tuple(self.graph.predecessors(var))

                attr_dict = items

            elif isinstance(items, (list, tuple, np.ndarray)):
                assert (
                        len(items) == 3
                ), "The positional items container needs to contain exactly 3 items."

                if items[0] is not None:
                    parent_list = [par for par in self._filter_variable_names(items[0])]
                else:
                    parent_list = tuple(self.graph.predecessors(var))
                attr_dict = dict()
                if items[1] is not None:
                    attr_dict.update({self.function_key: items[1]})
                if items[2] is not None:
                    attr_dict.update({self.noise_key: items[2]})

            else:
                raise ValueError(
                    f"Intervention items container '{type(items)}' not supported."
                )

            if var not in self.interventions_backup_attr:
                # the variable has NOT already been backed up. If we overwrite it now it is fine. If it had already been
                # in the backup, then it we would need to pass on this step, in order to not overwrite the backup
                # (possibly with a previous intervention)
                self.interventions_backup_attr[var] = deepcopy(self.graph.nodes[var])

            if var not in self.interventions_backup_parent:
                # the same logic goes for the parents backup.
                parent_backup = []
                for parent in list(self.graph.predecessors(var)):
                    parent_backup.append(parent)
                    self.graph.remove_edge(parent, var)
                self.interventions_backup_parent[var] = parent_backup

            self.graph.add_node(var, **attr_dict)
            for parent in parent_list:
                self.graph.add_edge(parent, var)

    def do_intervention(self, variables: Collection, values: Collection[float]):
        """
        Perform do-interventions, i.e. setting specific variables to a constant value.
        This method doesn't change the current noise models.

        Convenience wrapper around ``interventions`` method.

        Parameters
        ----------
        variables : Collection,
            the variables to intervene on.
        values : Collection[float],
            the constant values the chosen variables should be set to.

        Returns
        -------
            None
        """
        interventions_dict = dict()
        for var, val in zip(variables, values):
            interventions_dict[var] = ([], LinearAssignment(1, val), None)
        self.intervention(interventions_dict)

    def soft_intervention(self, variables: Collection, noise_models: Collection[float]):
        """
        Perform hard interventions, i.e. setting specific variables to a constant value.
        This method doesn't change the current noise models.

        Convenience wrapper around ``interventions`` method.

        Parameters
        ----------
        variables : Collection,
            the variables to intervene on.
        noise_models : Collection[float],
            the constant values the chosen variables should be set to.

        Returns
        -------
            None
        """
        interventions_dict = dict()
        for var, noise in zip(variables, noise_models):
            interventions_dict[var] = (None, None, noise)
        self.intervention(interventions_dict)

    def undo_intervention(self, variables: Union[List, Tuple, np.ndarray] = None):
        """
        Method to undo previously done interventions.

        The variables whose interventions should be made undone can be provided in the ``variables`` argument. If no
        list is supplied, all interventions will be undone.
        :param variables: list-like, the variables to be undone.
        """
        if variables is not None:
            present_variables = self._filter_variable_names(variables)
        else:
            present_variables = list(self.interventions_backup_attr.keys())

        for var in present_variables:
            try:
                attr_dict = self.interventions_backup_attr.pop(var)
                parents = self.interventions_backup_parent.pop(var)
            except KeyError:
                logging.warning(
                    f"Variable '{var}' not found in intervention backup. Omitting it."
                )
                continue

            self.graph.add_node(var, **attr_dict)
            for parent in list(self.graph.predecessors(var)):
                self.graph.remove_edge(parent, var)
            for parent in parents:
                self.graph.add_edge(parent, var)

    def reseed(self, seed: int):
        for var in self.get_variables():
            self.graph.nodes[var][self.noise_key].set_seed(seed)

    def plot(
        self,
        draw_labels: bool = True,
        node_size: int = 500,
        figsize: Tuple[int, int] = (6, 4),
        dpi: int = 150,
        alpha: float = 0.5,
        **kwargs,
    ):
        """
        Plot the causal graph of the scm in a dependency oriented way.

        This will attempt a tree plot of the scm, in the case that the graph is indeed a tree.
        However, because a causal graph is a DAG and can thus have directionless cycles (but not directional cycles), a tree
        structure often can't be computed. Therefore this method relies on graphviz to compute a
        feasible representation of the causal graph.

        The graphviz package has been marked as an optional package for this module and therefore needs to be installed
        by the user.
        Note, that (at least on Ubuntu) graphviz demands further libraries to be supplied, thus the following
        command will install the necessary dependencies, if graphviz couldn't be found on your system.
        Open a terminal and type:

            ``sudo apt-get install graphviz libgraphviz-dev pkg-config``
        Parameters
        ----------
        draw_labels : (optional) bool,
            Whether to draw the node labels onto the node. Can look unwieldy if the names are long.
            Default is True.
        node_size : (optional) int,
            the size of the node circles in the graph. Bigger values mean bigger circles.
            Default is 500.
        figsize : (optional) tuple,
            the size of the figure to be passed to matplotlib. Default is (6, 4).
        dpi : (optional) int,
            the dots per inch arg for matplotlib. Default is 150.
        alpha : (optional) float,
            the statistical significance level for the test. Default value is 0.05.
        kwargs :
            arguments to be passed to the ``networkx.draw`` method. Check its documentation for a full list.
        """
        if nx.is_tree(self.graph):
            pos = self._hierarchy_pos(root=self.roots)
        else:
            pos = graphviz_layout(self.graph, prog="dot")
        plt.title(self.scm_name)
        if draw_labels:
            labels = self.var_names_draw_dict
        else:
            labels = {}
        plt.figure(figsize=figsize, dpi=dpi)
        nx.draw(
            self.graph,
            pos=pos,
            labels=labels,
            with_labels=True,
            node_size=node_size,
            alpha=alpha,
            **kwargs,
        )

    def str(self):
        """
        Computes a string representation of the assignment functions for each variable and also mentions on which
        variables an intervention has been applied.
        :return: str, the representation.
        """
        lines = [
            f"Structural Causal Model of {self.nr_variables} variables: "
            + ", ".join(self.var_names),
            f"Following variables have been intervened on: {list(self.interventions_backup_attr.keys())}",
            "Current Assignment Functions are:",
        ]
        max_var_space = max([len(var_name) for var_name in self.var_names])
        for node in self.graph.nodes:
            parents_vars = [pred for pred in self.graph.predecessors(node)]
            line = f"{str(node).rjust(max_var_space)} := {self.graph.nodes[node][self.function_key].str(parents_vars)}"
            lines.append(line)
        return "\n".join(lines)

    def get_variables(self, causal_order=True):
        if causal_order:
            return self._causal_iterator()
        else:
            return self.graph.nodes

    def _build_graph(self, assignment_map):
        for node_name, (parents_list, function, noise_model) in assignment_map.items():
            self.graph.add_node(
                node_name, **{self.function_key: function, self.noise_key: noise_model}
            )
            if parents_list:
                for parent in parents_list:
                    self.graph.add_edge(parent, node_name)
            else:
                self.roots.append(node_name)

    def _filter_variable_names(self, variables: Iterable):
        """
        Filter out variable names, that are not currently in the graph. Warn for each variable that wasn't present.

        Returns a generator which iterates over all variables that have been found in the graph.

        :param variables: list, the variables to be filtered
        :return: generator, generates the filtered variables in sequence.
        """
        for variable in variables:
            if variable in self.graph.nodes:
                yield variable
            else:
                logging.warning(
                    f"Variable '{variable}' not found in graph. Omitting it."
                )

    def _causal_iterator(self, variables: Iterable = None):
        """
        Provide a causal iterator through the graph starting from the roots going to the variables needed.

        This iterator passes only the ancestors of the variables and thus is helpful in filtering out all the variables
        that have no causal effect on the desired variables.

        :param variables: list, the names of all the variables that are to be considered. Names that cannot be found
        in the naming list of the graph will be ignored (warning raised).
        :return: iterator, a generator object giving back all the ancestors.
        """
        if variables is None:
            for node in nx.topological_sort(self.graph):
                yield node
            return
        visited_nodes: Set = set()
        vars_causal_priority: Dict = defaultdict(int)
        queue = deque([var for var in self._filter_variable_names(variables)])
        while queue:
            nn = queue.popleft()
            if nn not in visited_nodes:
                for parent in self.graph.predecessors(nn):
                    vars_causal_priority[parent] = max(
                        vars_causal_priority[parent], vars_causal_priority[nn] + 1
                    )
                    queue.append(parent)
                visited_nodes.add(nn)
        for key, _ in sorted(
            vars_causal_priority.items(), key=lambda x: x[1], reverse=True
        ):
            yield key

    def _hierarchy_pos(
        self, root=None, width=1.0, vert_gap=0.2, vert_loc=0, leaf_vs_root_factor=0.5
    ):

        """
        If the graph is a tree this will return the positions to plot this in a
        hierarchical layout.

        Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
        but with some modifications.

        We include this because it may be useful for plotting transmission trees,
        and there is currently no networkx equivalent (though it may be coming soon).

        There are two basic approaches we think of to allocate the horizontal
        location of a node.

        - Top down: we allocate horizontal space to a node.  Then its ``k``
          descendants split up that horizontal space equally.  This tends to result
          in overlapping nodes when some have many descendants.
        - Bottom up: we allocate horizontal space to each leaf node.  A node at a
          higher level gets the entire space allocated to its descendant leaves.
          Based on this, leaf nodes at higher levels get the same space as leaf
          nodes very deep in the tree.

        We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
        determining how much of the horizontal space is based on the bottom up
        or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
        down.


        Parameters
        ----------
        **G** the graph (must be a tree)

        **root** the root node of the tree
        - if the tree is directed and this is not given, the root will be found and used
        - if the tree is directed and this is given, then the positions will be
          just for the descendants of this node.
        - if the tree is undirected and not given, then a random choice will be used.

        **width** horizontal space allocated for this branch - avoids overlap with other branches

        **vert_gap** gap between levels of hierarchy

        **vert_loc** vertical location of root

        **leaf_vs_root_factor**

        xcenter: horizontal location of root
        """
        G = self.graph
        if not nx.is_tree(G):
            raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(
                    iter(nx.topological_sort(G))
                )  # allows back compatibility with nx version 1.11
            else:
                root = np.random.choice(list(G.nodes))

        def __hierarchy_pos(
            G,
            root,
            leftmost,
            width,
            leafdx=0.2,
            vert_gap=0.2,
            vert_loc=0,
            xcenter=0.5,
            rootpos=None,
            leafpos=None,
            parent=None,
        ):
            """
            see hierarchy_pos docstring for most arguments

            pos: a dict saying where all nodes go if they have been assigned
            parent: parent of this branch. - only affects it if non-directed

            """

            if rootpos is None:
                rootpos = {root: (xcenter, vert_loc)}
            else:
                rootpos[root] = (xcenter, vert_loc)
            if leafpos is None:
                leafpos = {}
            children = list(G.neighbors(root))
            leaf_count = 0
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                rootdx = width / len(children)
                nextx = xcenter - width / 2 - rootdx / 2
                for child in children:
                    nextx += rootdx
                    rootpos, leafpos, newleaves = __hierarchy_pos(
                        G,
                        child,
                        leftmost + leaf_count * leafdx,
                        width=rootdx,
                        leafdx=leafdx,
                        vert_gap=vert_gap,
                        vert_loc=vert_loc - vert_gap,
                        xcenter=nextx,
                        rootpos=rootpos,
                        leafpos=leafpos,
                        parent=root,
                    )
                    leaf_count += newleaves

                leftmostchild = min(
                    (x for x, y in [leafpos[child] for child in children])
                )
                rightmostchild = max(
                    (x for x, y in [leafpos[child] for child in children])
                )
                leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
            else:
                leaf_count = 1
                leafpos[root] = (leftmost, vert_loc)
            #        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
            print(leaf_count)
            return rootpos, leafpos, leaf_count

        xcenter = width / 2.0
        if isinstance(G, nx.DiGraph):
            leafcount = len(
                [node for node in nx.descendants(G, root) if G.out_degree(node) == 0]
            )
        elif isinstance(G, nx.Graph):
            leafcount = len(
                [
                    node
                    for node in nx.node_connected_component(G, root)
                    if G.degree(node) == 1 and node != root
                ]
            )
        rootpos, leafpos, leaf_count = __hierarchy_pos(
            G,
            root,
            0,
            width,
            leafdx=width * 1.0 / leafcount,
            vert_gap=vert_gap,
            vert_loc=vert_loc,
            xcenter=xcenter,
        )
        pos = {}
        for node in rootpos:
            pos[node] = (
                leaf_vs_root_factor * leafpos[node][0]
                + (1 - leaf_vs_root_factor) * rootpos[node][0],
                leafpos[node][1],
            )
        xmax = max(x for x, y in pos.values())
        for node in pos:
            pos[node] = (pos[node][0] * width / xmax, pos[node][1])
        return pos
