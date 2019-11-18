from functools import reduce

from scm import *
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def chance_drop_per_level(level, max_chance):
    return max_chance / (10 ** level)


def alternating_signs(length):
    return np.array([-1, 1] * (length // 2) if length % 2 == 0 else ([-1, 1] * (length // 2 + 1))[:length])


def simulate(
        nr_genes: int = 500,
        master_genes: Union[int, np.ndarray, List] = None,
        levels_of_dependency: int = 5,
        max_connection_chance: float = 0.9,
        chance_per_level_func=chance_drop_per_level,
):
    # =====================
    # PARAMETERS
    # --------------------
    gene_names = np.array([r"G_" + str(i) + "" for i in range(nr_genes)])
    if master_genes is None:
        master_genes = gene_names[0:3]
    elif isinstance(master_genes, int):
        master_genes = gene_names[0:master_genes]
    elif isinstance(master_genes, np.ndarray):
        if master_genes.dtype not in [bool, int]:
            raise ValueError("'master_genes' parameter dtype needs to be boolean or integer.")
        master_genes = gene_names[master_genes]
    else:
        raise ValueError("'master_genes' parameter type not supported.")

    nr_dep_levels = max(min(nr_genes, 5), int(np.log(nr_genes)))

    # =====================

    population_per_level = np.diff((np.power(np.linspace(0, 1, nr_dep_levels), 2) * nr_genes)).astype(int)
    population_per_level = np.array([len(master_genes)] + population_per_level.tolist())
    population_per_level[-1] += nr_genes - population_per_level.sum()
    chance_per_level = {level: chance_per_level_func(level, max_connection_chance) for level in range(nr_dep_levels)}

    genes_to_level = dict()
    levels_to_genes: dict = defaultdict(list)
    curr_offset = 0
    for level, amount_genes in enumerate(population_per_level):
        for name in gene_names[curr_offset:curr_offset + amount_genes]:
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

        for this_level in range(gene_level - 1, -1, - 1):
            if this_level in levels_to_genes:
                parent_pool = levels_to_genes[this_level]
                parent_mask = np.random.binomial(1, chance_per_level[this_level], size=len(parent_pool)).astype(bool)
                parents[this_level] = parent_pool[parent_mask]
                nr_coeffs = len(parents[this_level])
                signs = alternating_signs(nr_coeffs)
                coeffs += [1. * np.random.rand(nr_coeffs) * signs]

        if coeffs:
            coeffs = np.concatenate(coeffs)

        assignment_dict[gene] = [reduce(lambda x, y: x + y.tolist(), parents.values(), []),
                                 LinearAssignment(noise_coeff, offset, *coeffs),
                                 NoiseGenerator("normal",
                                                loc=0, scale=np.random.rand() * 1)]

    gene_tex_names = {name: r"$G_{" + str(i) + "}$" for name, i in zip(gene_names, range(nr_genes))}
    cn = SCM(assignment_dict, variable_tex_names=gene_tex_names)
    return cn


def analyze_distributions(
        scm_net,
        sample=None,
        genes=None,
        figsize=(20, 20),
        bins=50
):
    if genes is None:
        genes = [g for i, g in enumerate(scm_net.get_variables()) if i < 100]
    if sample is None:
        rs = np.random.RandomState()
        sample = scm_net.sample(10000)
        sample = pd.DataFrame(rs.poisson(np.exp(sample)), columns=sample.columns)

    sample[genes].hist(bins=bins, figsize=figsize)
    plt.show()

    def quadr_poly(mu, phi):
        return phi * np.power(mu, 2) + mu

    mean = sample.mean(axis=0)
    var = sample.var(axis=0)
    plt.scatter(mean, var, color="black")
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    popt, _ = curve_fit(
        quadr_poly,
        mean,
        var
    )
    mean_sorted = np.sort(mean)
    plt.plot(mean_sorted, quadr_poly(mean_sorted, *popt), color="red")
    plt.title("Mean-Variance-Relationship")
    plt.show()


if __name__ == '__main__':

    causal_net = simulate(10, 2)
    print(causal_net)
    causal_net.plot(False, node_size=50, alpha=0.5)
    analyze_distributions(scm_net=causal_net)


