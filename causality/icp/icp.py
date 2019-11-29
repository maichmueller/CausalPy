import itertools as it
from copy import deepcopy
from typing import *

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
from abc import ABC, abstractmethod


class ICP(ABC):
    @abstractmethod
    def infer(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        envs: np.ndarray,
        target_variable: Union[int, str],
        alpha: float = 0.05,
        *args,
        **kwargs,
    ):
        pass


class LinICP(ICP):
    def __init__(
        self,
        alpha: float = 0.05,
        fit_intercept: bool = True,
        residual_test: str = "normal",
        filter_variables: bool = True,
        filter_method: str = "lasso_sqrt",
        ignored_subsets: Optional[Set] = None,
        nr_parents_limit: Optional[int] = None,
    ):
        """

        Parameters
        ----------
        alpha
        fit_intercept
        residual_test
        filter_variables
        filter_method
        ignored_subsets : (optional) Set
            Specific subsets of variables, that are not to be checked for causal parents.
            The method will still check the variables occurring in these subsets in other combinations!
        nr_parents_limit : (optional) int
            The upper limiting number of causal parents to consider. If not given, the method will search in all
            possible, not excluded subsets.
        """

        self.fit_intercept = fit_intercept
        self.residual_test = residual_test
        self.filter_variables = filter_variables
        self.filter_method = filter_method
        self.ignored_subsets: Optional[Set] = ignored_subsets
        self.nr_parents_limit: Optional[int] = nr_parents_limit

        self.n: int = -1
        self.p: int = -1
        self.alpha: float = alpha
        self.target_name: object = None
        self.index_to_varname: pd.Series = pd.Series([])
        self.varname_to_index: pd.Series = pd.Series([])
        self.variables: np.ndarray = np.array([])
        self.accepted_sets: Set = set()

    def filter_candidates(self, obs, target, nr_iter=100):
        if self.filter_method == "lasso_sqrt":
            _, filtered, _ = sklearn.linear_model.lars_path(
                obs, target, method="lasso", max_iter=nr_iter, return_path=False
            )
        else:
            raise NotImplementedError(f"Method for filtering type '{self.filter_method}' not yet implemented.")
        return filtered

    def infer(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        envs: np.ndarray,
        target_variable: Union[int, str],
        alpha: Optional[float] = None,
        *args,
        **kwargs,
    ):
        r"""
        Perform Linear Invariant Causal Prediction (ICP) on the data provided on object construction.
        This method assumes the data stems from a linear gaussian SCM, i.e.
         - all assignment functions are linear in their parents and noise variables,
         - the residuals follow a gaussian distribution.

        For details refer to [1]_.

        Parameters
        ----------
        alpha : float
            Significance level of the test. P(\hat{S} \subset S^*) \gte 1-`alpha`
        obs : (n, p) ndarray or DataFrame
            The data of all environment observations from the variables of interest.
        envs : (n,) ndarray
            Array of environment indices for the observation dataset.
        target_variable : int or ``obs`` DataFrame column accessor
            The target variable to perform causal parent identification on.


        Returns
        -------
        tuple
            The identified causal parent set, \hat{S}, as tuple of variable names.

        References
        ----------
        [1] J. Peters, P. Bühlmann, N. Meinshausen:
        Causal inference using invariant prediction: identification and confidence intervals, arXiv:1501.01332,
        Journal of the Royal Statistical Society, Series B (with discussion) 78(5):947-1012, 2016.
        """
        if alpha is None:
            alpha = self.alpha
        self.n, self.p = obs.shape[0], obs.shape[1] - 1
        assert (
            len(envs) == self.n
        ), f"Number of observation samples ({len(envs)}) and number of environment labels ({self.n}) have to be equal."

        self.alpha = alpha
        self.target_name = target_variable

        if isinstance(obs, pd.DataFrame):
            self.variables = obs.columns.values
            target = obs[target_variable].to_numpy().flatten()
            obs = obs.drop(columns=[target_variable])
            self.index_to_varname = pd.Series(
                obs.columns, index=range(self.p)
            )
            self.varname_to_index = pd.Series(
                range(self.p), index=obs.columns
            )
            obs = obs.to_numpy()

        elif isinstance(obs, np.ndarray):
            self.index_to_varname = pd.Series(np.arange(self.p))
            self.variables = self.index_to_varname.values
            target = obs[:, target_variable].flatten()
            obs = np.delete(obs, target_variable, axis=1)
            itv = self.index_to_varname
            target_index = itv[itv == target_variable].index
            self.varname_to_index = (itv.loc[target_index + 1 :] - 1).drop(index=target_variable)
        else:
            raise ValueError(
                "Observations have to be either a pandas DataFrame or numpy ndarray."
            )

        if self.filter_variables:
            pre_filtered_vars = np.sort(
                self.filter_candidates(
                    obs, target, kwargs.pop("nr_iter", 100)
                )
            )
            obs = obs[:, pre_filtered_vars]
            p = obs.shape[1]

            self.index_to_varname = self.index_to_varname.reindex(
                pre_filtered_vars
            ).reset_index(drop=True)

            self.varname_to_index = pd.Series(
                self.index_to_varname.index, index=self.index_to_varname.values
            )
        else:
            p = self.p

        environments: Dict = {env: envs == env for env in np.unique(envs)}

        subset_iterator = self._subset_iterator(
            elements=self.index_to_varname.values,
            rejected_subsets=self.ignored_subsets,
            nr_parents_limit=self.nr_parents_limit,
        )
        p_values_per_elem = np.zeros(p)

        subset, finished = next(subset_iterator)
        while not finished:
            subset_indices = self.varname_to_index[list(subset)].values
            p_value = self._test_plausible_parents(obs, target, environments, subset_indices)
            print("tested:", subset, "\t p value:", p_value)
            if subset:
                # this if condition excludes the test case of the empty set
                p_values_per_elem[subset_indices] = np.maximum(
                    p_values_per_elem[subset_indices], p_value
                )
            subset, finished = subset_iterator.send(p_value <= self.alpha)
        # once the routine has finished the variable subset will hold the latest estimate of the causal parents

        p_values_per_elem = pd.Series(
            1 if not subset else p_values_per_elem,
            index=[self.index_to_varname[i] for i in np.arange(p)],
        )  # add variable names information to the data
        return subset, p_values_per_elem

    @staticmethod
    def test_residuals(sample1, sample2, test="normal", **kwargs):
        """
        Test for the equality of the distribution of input 1 versus 2.

        This method performs a different test depending on the ``test`` parameter. See the parameter documentation for
        more information.

        Parameters
        ----------
        sample1 : (n, p) np.ndarray,
            The data pertaining to the first gaussian.
        sample2 : (n, p) np.ndarray,
            The data pertaining to the second gaussian.
        test : "normal", "ranks", "ks" or Callable,
            For 'normal' a t-test on the mean and an F-test on the variance are performed.
            For 'ranks' the F-test is switched with Levene's test on equal variance.
            For 'ks' a Kolmogorov-Smirnov test on equal distribution is applied.
            For a Callable, the method is called with 'test(sample1, sample2)' and assumed to return a p-value.
        kwargs: kwargs,
            Arguments passable to the scipy.stats.levene, Kolomogorov-Smirnov or callable test method, depending on
            which has been chosen. More information can be found in [1]_ and [2]_.

        Returns
        -------
        tuple
            A 2-tuple of the test p-values of the equal mean test and the equal variance test.

        References
        ----------
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
        [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
        [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        """
        if test == "ranks":
            p_equal_mean = scipy.stats.ttest_ind(
                sample1, sample2, equal_var=False
            ).pvalue
            #  the levene test is apparently more robust than the standard F-test given the sensitivity of the F-test
            # to non-normality of the data.
            p_equal_var = scipy.stats.levene(sample1, sample2, **kwargs).pvalue
            p_val = 2 * min(p_equal_mean, p_equal_var)

        elif test == "ks":
            p_val = scipy.stats.ks_2samp(sample1, sample2, **kwargs).pvalue

        elif isinstance(test, Callable):
            p_val = test(sample1, sample2, **kwargs)

        elif test == "normal":
            n1 = len(sample1)
            n2 = len(sample2)
            mean1 = np.mean(sample1)
            mean2 = np.mean(sample2)
            # ddof = 0 as per numpy doc: "ddof = 0 provides the ML estimation of variance for normally distributed
            # variables"
            std1 = np.std(sample1, ddof=0)
            std2 = np.std(sample2, ddof=0)
            p_equal_mean = scipy.stats.ttest_ind_from_stats(
                mean1, std1, n1, mean2, std2, n2, equal_var=False
            ).pvalue
            f_test = (std1 ** 2) / (std2 ** 2)
            p_equal_var = 2 * min(
                scipy.stats.f.cdf(f_test, n1 - 1, n2 - 1),
                scipy.stats.f.sf(f_test, n1 - 1, n2 - 1),
            )
            p_val = 2 * min(p_equal_mean, p_equal_var)

        else:
            raise ValueError(f"Test input parameter '{test}' not supported.")

        return p_val

    def _test_plausible_parents(self, obs, target: np.ndarray, envs: Dict, s: Union[np.ndarray, List, Tuple], ):
        if not len(s):
            obs_S = np.ones((self.n, 1))
        else:
            obs_S = sklearn.preprocessing.add_dummy_feature(obs[:, s])
        lr = sklearn.linear_model.LinearRegression(fit_intercept=True)
        lr.fit(obs_S, target)
        residuals = lr.predict(obs_S) - target
        p_value = 1
        # the paper suggests to test the residual of data in an environment e against the
        # the residuals of the data not in e.
        # TODO: find out, whether this isn't equivalent to the slightly faster method of testing the residuals of
        # TODO: each environment e against environment e + 1.
        for env in envs:
            env_indices = envs[env]
            p_value_update = self.test_residuals(
                residuals[env_indices],
                residuals[np.logical_not(env_indices)],
                test=self.residual_test,
            )
            p_value = min(p_value, p_value_update)
        return p_value * len(envs)  # Bonferroni correction for p value

    @staticmethod
    def _subset_iterator(
        elements: Union[int, Collection, Set],
        candidates: Set[Tuple] = None,
        rejected_subsets: Set[Tuple] = None,
        nr_parents_limit: int = None,
    ):
        if isinstance(elements, int):
            elements = set(range(elements))
        elif isinstance(elements, Collection):
            elements = set(elements)
        else:
            raise ValueError(
                f"Expected 'elements' to be a sequence of identifiers. "
                f"Got {type(elements)} instead."
            )

        if rejected_subsets is None:
            rejected_subsets = set()
        if candidates is None:
            if not isinstance(elements, Set):
                candidates = set(elements)
            else:
                candidates = deepcopy(elements)
        else:
            candidates = set(candidates)

        if nr_parents_limit is None:
            nr_parents_limit = len(elements)

        for subset_uncut in it.chain.from_iterable(
            it.combinations(elements, len_subset)
            for len_subset in range(nr_parents_limit + 1)
        ):
            subset = candidates.intersection(subset_uncut)
            # converting to tuple here is necessary as sets are mutable, thus not hashable
            subset_tup = tuple(subset)
            if subset_tup in rejected_subsets:
                # the intersected subset has already been checked, moving on
                continue

            rejected = yield (subset_tup, False)

            if not rejected:
                candidates = subset
                if not candidates:
                    # empty candidates set means we have found no causal parent as there were at least two accepted
                    # subsets with empty intersection.
                    yield tuple(), True

            else:
                rejected_subsets.add(subset_tup)

        yield tuple(candidates), True
