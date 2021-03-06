from functools import reduce
from typing import Union, List

import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from scmodels import SCM
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def chance_drop_per_level(level, max_chance):
    return max_chance / (10 ** level)


def alternating_signs(length):
    return np.array(
        [1, -1] * (length // 2)
        if length % 2 == 0
        else ([1, -1] * (length // 2 + 1))[:length]
    )


def simulate(
    nr_genes: int = 500,
    master_genes: Union[int, np.ndarray, List] = None,
    max_connection_chance: float = 0.9,
    chance_per_level_func=chance_drop_per_level,
    seed=None,
):
    # =====================
    # PARAMETERS
    # --------------------
    rs = np.random.default_rng(seed)
    gene_names = np.array([r"G_" + str(i) + "" for i in range(nr_genes)])
    if master_genes is None:
        master_genes = gene_names[0:3]
    elif isinstance(master_genes, int):
        master_genes = gene_names[0:master_genes]
    elif isinstance(master_genes, np.ndarray):
        if master_genes.dtype not in [bool, int]:
            raise ValueError(
                "'master_genes' parameter dtype needs to be boolean or integer."
            )
        master_genes = gene_names[master_genes]
    else:
        raise ValueError("'master_genes' parameter type not supported.")

    nr_dep_levels = max(min(nr_genes, 5), int(np.log(nr_genes)))

    population_per_level = np.diff(
        (np.power(np.linspace(0, 1, nr_dep_levels), 2) * nr_genes)
    ).astype(int)
    population_per_level = np.array([len(master_genes)] + population_per_level.tolist())
    population_per_level[-1] += nr_genes - population_per_level.sum()
    chance_per_level = {
        level: chance_per_level_func(level, max_connection_chance)
        for level in range(nr_dep_levels)
    }

    # =====================

    genes_to_level = dict()
    levels_to_genes: dict = defaultdict(list)
    curr_offset = 0
    for level, amount_genes in enumerate(population_per_level):
        for name in gene_names[curr_offset : curr_offset + amount_genes]:
            genes_to_level[name] = level
            levels_to_genes[level].append(name)
        curr_offset += amount_genes
    for level, genes in levels_to_genes.items():
        levels_to_genes[level] = np.array(genes)

    assignment_dict = dict()

    gene_names_pbar = tqdm(gene_names)
    for gene in gene_names_pbar:
        gene_names_pbar.set_description(f"Processing {gene}")
        gene_level = genes_to_level[gene]

        parents = dict()

        # defining the coefficients for the linear function.
        # offset is a random number in [0, 20]
        # noise coefficient is always 1
        # coefficients for parents are chosen at random from [0, 2]
        offset = 0
        noise_coeff = 1
        coeffs = []

        for this_level in range(gene_level - 1, -1, -1):
            if this_level in levels_to_genes:
                parent_pool = levels_to_genes[this_level]
                parent_mask = rs.binomial(
                    1, chance_per_level[this_level], size=len(parent_pool)
                ).astype(bool)
                parents[this_level] = parent_pool[parent_mask]
                nr_coeffs = len(parents[this_level])
                signs = alternating_signs(nr_coeffs)
                coeffs.append(
                    rs.random(nr_coeffs)
                    * rs.choice([1, 0.5, 2], p=[0.8, 0.1, 0.1])
                    * signs
                )

        if coeffs:
            coeffs = np.concatenate(coeffs)

        assignment_dict[gene] = [
            reduce(lambda x, y: x + y.tolist(), parents.values(), []),
            f"{offset} + {noise_coeff} * N + " + " + ".join(coeffs),
            NoiseGenerator(
                "skewnorm",
                scale=rs.random() * (nr_dep_levels - gene_level),
                a=0.2,
                source="scipy",
            ),
        ]

    gene_tex_names = {
        name: r"$G_{" + str(i) + "}$" for name, i in zip(gene_names, range(nr_genes))
    }
    cn = SCM(assignment_dict, variable_tex_names=gene_tex_names)
    if seed is not None:
        cn.seed(seed + 10)
    return cn


def to_count(sample):
    rs = np.random.RandomState()
    return pd.DataFrame(
        rs.poisson(
            torch.nn.Softplus(beta=1)(torch.as_tensor(sample.to_numpy())).numpy()
        ),
        columns=sample.columns,
    )


def analyze_distributions(scm_net, sample=None, genes=None, figsize=(20, 20), bins=50):
    if genes is None:
        genes = [g for i, g in enumerate(scm_net.get_variables()) if i < 100]
    if sample is None:
        rs = np.random.RandomState()
        sample = scm_net.sample(10000)
        sample = pd.DataFrame(
            rs.poisson(
                torch.nn.Softplus(beta=1)(torch.as_tensor(sample.to_numpy())).numpy()
            ),
            columns=sample.columns,
        )

    sample[genes].hist(bins=bins, figsize=figsize)
    plt.show()

    def neg_binomial(mu, phi):
        return phi * np.power(mu, 2) + mu

    def dropout_rate_nb(mu, phi):
        phi_min1 = 1 / phi
        return np.power(phi_min1 / (mu + phi_min1), phi_min1)

    def zinb(mu, pi, alpha):
        return (1 - pi) * mu * (mu + (pi + alpha) * mu)

    def dropout_rate_zinb(mu, pi, alpha):
        return pi + (1 - pi) * np.power((1 / (1 + alpha * mu)), 1 / alpha)

    mean = sample.mean(axis=0)
    var = sample.var(axis=0)

    fig = plt.figure(1, figsize=(10, 8))

    plt.scatter(mean, var, color="black", alpha=0.1)
    mean_sorted = mean.sort_values()
    plt.plot(
        mean_sorted, mean_sorted, color="blue", label=r"Poi: Var$(\mu) = \mu$",
    )

    popt_nb = curve_fit(neg_binomial, mean, var)[0].flatten()
    mean_sorted = mean.sort_values()
    plt.plot(
        mean_sorted,
        neg_binomial(mean_sorted, *popt_nb),
        color="red",
        label=f"NB~: Var$(\mu, \phi) = \mu + \phi \mu^2$ \t [$\phi = {popt_nb.round(3)[0]}$]",
    )

    popt_zinb = curve_fit(zinb, mean, var)[0].flatten()
    plt.plot(
        mean_sorted,
        zinb(mean_sorted, *popt_zinb),
        color="green",
        label=r"ZINB: Var$(\mu, \pi, \alpha) = (1 - \pi)\mu (\mu + (\pi + \alpha) \mu)$"
        + "\t "
        + r"[$\pi"
        + f" = {popt_zinb.round(3)[0]}, "
        + r"\alpha"
        + f" = {popt_zinb.round(3)[1]}$]",
    )
    plt.legend(fancybox=True, shadow=True)
    plt.loglog()
    plt.xlabel("Mean $\mu$")
    plt.ylabel("Variance")
    plt.title("Mean-Variance-Relationship")
    plt.show()

    #########################
    # Expected dropout rate #
    #########################

    obs_dropout_per_gene = ((sample == 0).sum(axis=0) / sample.shape[0])[
        mean_sorted.index.values
    ]

    fig = plt.figure(2, figsize=(10, 8))

    plt.scatter(mean_sorted, obs_dropout_per_gene, color="black", alpha=0.1)
    plt.plot(
        mean_sorted,
        dropout_rate_nb(mean_sorted, *popt_nb),
        color="red",
        label=r"NB~: $\mathbb{P}(0 \mid| \mu, \phi) = \left(\frac{1}{1 + \phi \mu}\right)^{\phi^{-1}}$"
        + "\t"
        + f"[$\phi = {popt_nb.round(3)[0]}$]",
    )
    plt.plot(
        mean_sorted,
        dropout_rate_zinb(mean_sorted, *popt_zinb),
        color="green",
        label=r"ZINB: $\mathbb{P}(0 \mid| \mu, \pi, \alpha) = \pi  + (1 - \pi) * "
        r"\left(\frac{1}{1 + \alpha \mu}\right)^{\alpha^{-1}}$"
        + "\t"
        + f"[$\pi = {popt_zinb.round(3)[0]}, "
        + r"\alpha"
        + f" = {popt_zinb.round(3)[1]}$]",
    )
    plt.legend(fancybox=True, shadow=True)
    plt.xscale("log")
    plt.xlabel("Mean $\mu$")
    plt.ylabel("$\mathbb{P}(0 \mid \mu)$")
    plt.title("Mean-Dropout-Relationship")
    plt.show()
    return sample


if __name__ == "__main__":
    nr_genes = 20000
    causal_net = simulate(nr_genes, 2, seed=1)
    print(causal_net)
    # causal_net.plot(False, node_size=50, alpha=0.5)
    # plt.show()
    sample = analyze_distributions(scm_net=causal_net)
    sample_var = sample.var().sort_values()
    sample.to_csv(f"sc_data_{nr_genes}.csv", index=False)
