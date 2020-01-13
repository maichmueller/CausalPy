import os
from collections import namedtuple
from typing import Union, Optional, Callable

import numpy as np
import torch
import pandas as pd
import networkx as nx
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .icpbase import ICPredictor
from causalpy.neural_networks import AgnosticModel, Linear3D, MatrixSampler


Hyperparams = namedtuple("Hyperparams", "alpha beta gamma")


class AgnosticPredictor(ICPredictor):
    def __init__(
            self,
            network: Optional[torch.nn.Module] = None,
            epochs: int = 100,
            batch_size: int = 1024,
            loss_transform_res_to_par: str = "sum",
            loss_transform_res_to_res: str = "sum",
            compare_residuals_pairwise: bool = True,
            residual_equality_measure: Union[str, Callable] = "mmd",
            variable_independence_metric: str = "hsic",
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            hyperparams: Optional[Hyperparams] = None,
            log_level: bool = True,
    ):
        super().__init__(log_level=log_level)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.epochs = epochs
        self.network = network if network is not None else AgnosticModel()
        self.hyperparams = (
            Hyperparams(alpha=1, beta=1, gamma=1)
            if hyperparams is None
            else hyperparams
        )

        self.optimizer = (
            torch.optim.Adam(network.parameters(), lr=1e-3)
            if optimizer is None
            else optimizer
        )

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.9
            )

        # if loss_transform_res_to_par == "sum":
        #     self.loss_reduce_par = torch.sum
        # elif loss_transform_res_to_par == "max":
        #     self.loss_reduce_par = torch.max
        # elif loss_transform_res_to_par == "alphamoment":
        #     self.loss_reduce_par = partial(
        #         self._alpha_moment, alpha=kwargs.pop("alpha", 1)
        #     )
        # else:
        #     args = ['sum', 'max', 'alphamoment']
        #     raise ValueError(
        #         f"Loss Transform function needs to be one of {args}. Provided: '{loss_transform_res_to_par}'."
        #     )
        #
        # if loss_transform_res_to_res == "sum":
        #     self.loss_reduce_res = torch.sum
        # elif loss_transform_res_to_res == "max":
        #     self.loss_reduce_res = torch.max
        # elif loss_transform_res_to_res == "alphamoment":
        #     self.loss_reduce_res = partial(
        #         self._alpha_moment, alpha=kwargs.pop("alpha", 1)
        #     )
        # else:
        #     args = ['sum', 'max', 'alphamoment']
        #     raise ValueError(
        #         f"Loss Transform function needs to be one of {args}. Provided: '{loss_transform_res_to_res}'."
        #     )
        #
        # self.residual_equality_measure_str = residual_equality_measure
        # if residual_equality_measure == "mmd":
        #     self.identical_distribution_metric = self.mmd_multiscale
        # elif residual_equality_measure == "moments":
        #     self.identical_distribution_metric = self.moments
        # elif isinstance(residual_equality_measure, Callable):
        #     self.identical_distribution_metric = residual_equality_measure
        # else:
        #     raise ValueError(
        #         f"Dependency measure '{residual_equality_measure}' not supported."
        #     )
        #
        # self.variable_independence_metric_str = variable_independence_metric
        # if variable_independence_metric == "hsic":
        #     self.variable_independence_metric = self.hsic

    def infer(
            self,
            obs: Union[pd.DataFrame, np.ndarray],
            envs: np.ndarray,
            target_variable: Union[int, str],
            alpha: float = 0.05,
            *args,
            **kwargs
            # skeleton=None,
            # train=5000, test=1000,
            # batch_size=-1, lr_gen=.001,
            # lr_disc=.01, lambda1=0.001, lambda2=0.0000001, nh=None, dnh=None,
            # verbose=True, losstype="fgan",
            # dagstart=0, dagloss=False,
            # dagpenalization=0.05, dagpenalization_increase=0.0,
            # linear=False, hlayers=2, idx=0):
    ):
        obs, target, environments = self.preprocess_input(obs, target_variable, envs)
        list_nodes = self.index_to_varname.index

        obs = obs.astype('float32')
        rows, cols = obs.shape()
        obs_tensor = torch.from_numpy(obs).to(self.device)

        conditional_net = None  # TODO: instantiate at this point a cINN or cVAE
        conditional_net.reset_parameters()
        optimizer = torch.optim.Adam(conditional_net.parameters(), lr=kwargs.pop("lr", 0.001))

        if losstype != "mse":
            discriminator = SAM_discriminator(cols, dnh, hlayers).to(device)
            discriminator.reset_parameters()
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_disc)
            criterion = torch.nn.BCEWitorchLogitsLoss()
        else:
            criterion = torch.nn.MSELoss()
            disc_loss = torch.zeros(1)

        graph_sampler = MatrixSampler(self.p, mask=skeleton, gumbel=False).to(self.device)
        graph_sampler.weights.data.fill_(2)
        graph_optimizer = torch.optim.Adam(graph_sampler.parameters(), lr=lr_gen)

        if not linear:
            neuron_sampler = MatrixSampler((nh, self.p), mask=False,
                                           gumbel=True).to(self.device)
            neuron_optimizer = torch.optim.Adam(list(neuron_sampler.parameters()),
                                                lr=lr_gen)

        _true = torch.ones(1).to(device)
        _false = torch.zeros(1).to(device)
        output = torch.zeros(nb_var, nb_var).to(device)

        noise = torch.randn(batch_size, nb_var).to(device)
        noise_row = torch.ones(1, nb_var).to(device)
        data_iterator = DataLoader(data, batch_size=batch_size,
                                   shuffle=True, drop_last=True)

        # RUN
        if verbose:
            pbar = tqdm(range(train + test))
        else:
            pbar = range(train + test)
        for epoch in pbar:
            for i_batch, batch in enumerate(data_iterator):
                g_optimizer.zero_grad()
                graph_optimizer.zero_grad()

                if losstype != "mse":
                    d_optimizer.zero_grad()

                if not linear:
                    neuron_optimizer.zero_grad()

                # Train torche discriminator

                if not epoch > train:
                    drawn_graph = graph_sampler()
                    if not linear:
                        drawn_neurons = neuron_sampler()
                    else:
                        drawn_neurons = None
                noise.normal_()
                generated_variables = sam(batch, noise,
                                          torch.cat([drawn_graph, noise_row], 0),
                                          drawn_neurons)

                if losstype == "mse":
                    gen_loss = criterion(generated_variables, batch)
                else:
                    disc_vars_d = discriminator(generated_variables.detach(), batch)
                    disc_vars_g = discriminator(generated_variables, batch)
                    true_vars_disc = discriminator(batch)

                    if losstype == "gan":
                        disc_loss = sum([criterion(gen, _false.expand_as(gen)) for gen in disc_vars_d]) / nb_var \
                                    + criterion(true_vars_disc, _true.expand_as(true_vars_disc))
                        # Gen Losses per generator: multiply py torche number of channels
                        gen_loss = sum([criterion(gen,
                                                  _true.expand_as(gen))
                                        for gen in disc_vars_g])
                    elif losstype == "fgan":

                        disc_loss = sum([th.mean(th.exp(gen - 1)) for gen in disc_vars_d]) / nb_var - th.mean(
                            true_vars_disc)
                        gen_loss = -sum([th.mean(th.exp(gen - 1)) for gen in disc_vars_g])

                    disc_loss.backward()
                    d_optimizer.step()

                filters = graph_sampler.get_proba()

                struc_loss = lambda1 * drawn_graph.sum()

                func_loss = 0 if linear else lambda2 * drawn_neurons.sum()
                regul_loss = struc_loss + func_loss

                if dagloss and epoch > train * dagstart:
                    dag_constraint = notears_constr(filters * filters)
                    loss = gen_loss + regul_loss + (dagpenalization +
                                                    (epoch - train * dagstart)
                                                    * dagpenalization_increase) * dag_constraint
                else:
                    loss = gen_loss + regul_loss
                if verbose and epoch % 20 == 0 and i_batch == 0:
                    pbar.set_postfix(gen=gen_loss.item() / cols,
                                     disc=disc_loss.item(),
                                     regul_loss=regul_loss.item(),
                                     tot=loss.item())

                if epoch < train + test - 1:
                    loss.backward(retain_graph=True)

                if epoch >= train:
                    output.add_(filters.data)

                g_optimizer.step()
                graph_optimizer.step()
                if not linear:
                    neuron_optimizer.step()

        return output.div_(test).cpu().numpy()

    class SAM(GraphModel):

        """SAM Algorithm.
        **Description:** Structural Agnostic Model is an causal discovery algorithm
        for DAG recovery leveraging both distributional asymetries and conditional
        independencies. the first version of SAM without DAG constraint is available
        as ``SAMv1``.
        **Data Type:** Continuous
        **Assumptions:** The class of generative models is not restricted with a
        hard contraint, but with soft constraints parametrized with the ``lambda1``
        and ``lambda2`` parameters, with gumbel softmax sampling. This algorithms greatly
        benefits from bootstrapped runs (nruns >=8 recommended).
        GPUs are recommended but not compulsory. The output is a DAG, but may need a
        thresholding as the output is averaged over multiple runs.
        Args:
            lr (float): Learning rate of the generators
            dlr (float): Learning rate of the discriminator
            lambda1 (float): L0 penalization coefficient on the causal filters
            lambda2 (float): L0 penalization coefficient on the hidden units of the
               neural network
            nh (int): Number of hidden units in the generators' hidden layers
               (regularized with lambda2)
            dnh (int): Number of hidden units in the discriminator's hidden layer
            train_epochs (int): Number of training epochs
            test_epochs (int): Number of test epochs (saving and averaging
               the causal filters)
            batch_size (int): Size of the batches to be fed to the SAM model.
               Defaults to full-batch.
            losstype (str): type of the loss to be used (either 'fgan' (default),
               'gan' or 'mse').
            hlayers (int): Defines the number of hidden layers in the discriminator.
            dagloss (bool): Activate the DAG with No-TEARS constraint.
            dagstart (float): Controls when the DAG constraint is to be introduced
               in the training (float ranging from 0 to 1, 0 denotes the start of
               the training and 1 the end).
            dagpenalisation (float): Initial value of the DAG constraint.
            dagpenalisation_increase (float): Increase incrementally at each epoch
               the coefficient of the constraint.
            linear (bool): If true, all generators are set to be linear generators.
            nruns (int): Number of runs to be made for causal estimation.
                   Recommended: >=8 for optimal performance.
            njobs (int): Numbers of jobs to be run in Parallel.
                   Recommended: 1 if no GPU available, 2*number of GPUs else.
            gpus (int): Number of available GPUs for the algorithm.
            verbose (bool): verbose mode
        .. note::
           Ref: Kalainathan, Diviyan & Goudet, Olivier & Guyon, Isabelle &
           Lopez-Paz, David & Sebag, Michèle. (2018). Structural Agnostic Modeling:
           Adversarial Learning of Causal Graphs.
        Example:
            >>> import networkx as nx
            >>> from cdt.causality.graph import SAM
            >>> from cdt.data import load_dataset
            >>> data, graph = load_dataset("sachs")
            >>> obj = SAM()
            >>> #The predict() method works without a graph, or with a
            >>> #directed or undirected graph provided as an input
            >>> output = obj.predict(data)    #No graph provided as an argument
            >>>
            >>> output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
            >>>
            >>> output = obj.predict(data, graph)  #With a directed graph
            >>>
            >>> #To view the graph created, run the below commands:
            >>> nx.draw_networkx(output, font_size=8)
            >>> plt.show()
        """

    def __init__(self, lr=0.01, dlr=0.01, lambda1=0.01, lambda2=0.00001, nh=200, dnh=200,
                 train_epochs=10000, test_epochs=1000, batchsize=-1,
                 losstype="fgan", dagstart=0.5, dagloss=True, dagpenalization=0,
                 dagpenalization_increase=0.001, linear=False, hlayers=2,
                 njobs=None, gpus=None, verbose=None, nruns=8):

        """Init and parametrize the SAM model."""
        super(SAM, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize
        self.losstype = losstype
        self.dagstart = dagstart
        self.dagloss = dagloss
        self.dagpenalization = dagpenalization
        self.dagpenalization_increase = dagpenalization_increase
        self.linear = linear
        self.hlayers = hlayers
        self.njobs = SETTINGS.get_default(njobs=njobs)
        self.gpus = SETTINGS.get_default(gpu=gpus)
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.nruns = nruns

    def predict(self, data, graph=None,
                return_list_results=False):
        """Execute SAM on a dataset given a skeleton or not.
        Args:
            data (pandas.DataFrame): Observational data for estimation of causal relationships by SAM
            skeleton (numpy.ndarray): A priori knowledge about the causal relationships as an adjacency matrix.
                      Can be fed either directed or undirected links.
        Returns:
            networkx.DiGraph: Graph estimated by SAM, where A[i,j] is the term
            of the ith variable for the jth generator.
        """
        if graph is not None:
            skeleton = th.Tensor(nx.adjacency_matrix(graph,
                                                     nodelist=list(data.columns)).todense())
        else:
            skeleton = None

        assert self.nruns > 0
        if self.gpus == 0:
            results = [run_SAM(data, skeleton=skeleton,
                               lr_gen=self.lr,
                               lr_disc=self.dlr,
                               verbose=self.verbose,
                               lambda1=self.lambda1, lambda2=self.lambda2,
                               nh=self.nh, dnh=self.dnh,
                               train=self.train,
                               test=self.test, batch_size=self.batchsize,
                               dagstart=self.dagstart,
                               dagloss=self.dagloss,
                               dagpenalization=self.dagpenalization,
                               dagpenalization_increase=self.dagpenalization_increase,
                               losstype=self.losstype,
                               linear=self.linear,
                               hlayers=self.hlayers,
                               device='cpu') for i in range(self.nruns)]
        else:
            results = parallel_run(run_SAM, data, skeleton=skeleton,
                                   nruns=self.nruns,
                                   njobs=self.njobs, gpus=self.gpus, lr_gen=self.lr,
                                   lr_disc=self.dlr,
                                   verbose=self.verbose,
                                   lambda1=self.lambda1, lambda2=self.lambda2,
                                   nh=self.nh, dnh=self.dnh,
                                   train=self.train,
                                   test=self.test, batch_size=self.batchsize,
                                   dagstart=self.dagstart,
                                   dagloss=self.dagloss,
                                   dagpenalization=self.dagpenalization,
                                   dagpenalization_increase=self.dagpenalization_increase,
                                   losstype=self.losstype,
                                   linear=self.linear,
                                   hlayers=self.hlayers)
        list_out = [i for i in results if not np.isnan(i).any()]
        try:
            assert len(list_out) > 0
        except AssertionError as e:
            print("All solutions contain NaNs")
            raise (e)
        W = sum(list_out) / len(list_out)
        return nx.relabel_nodes(nx.DiGraph(W),
                                {idx: i for idx,
                                            i in enumerate(data.columns)})

    def orient_directed_graph(self, *args, **kwargs):
        """Orient a (partially directed) graph."""
        return self.predict(*args, **kwargs)

    def orient_undirected_graph(self, *args, **kwargs):
        """Orient a undirected graph."""
        return self.predict(*args, **kwargs)

    def create_graph_from_data(self, *args, **kwargs):
        """Estimate a causal graph out of observational data."""
        return self.predict(*args, **kwargs)
