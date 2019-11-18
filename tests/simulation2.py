from functools import reduce

from scm import *
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from functools import partial


def population_perc(h):
    return h ** 2


def chance_modifier(level, max_chance):
    return max_chance / (10 ** level)


def alternating_signs(length):
    return np.array([-1, 1] * (length // 2) if length % 2 == 0 else ([-1, 1] * (length // 2 + 1))[:length])


if __name__ == '__main__':

    # =====================
    # PARAMETERS
    # --------------------
    nr_genes = 100

    gene_names = [r"G_" + str(i) + "" for i in range(nr_genes)]

    master_genes = gene_names[0:3]
    nr_dep_levels = max(min(nr_genes, 5), int(np.log(nr_genes)))

    max_connection_chance = 0.9

    levels_of_dependency = 5

    # =====================

    population_per_level = np.diff((np.power(np.linspace(0, 1, nr_dep_levels), 2) * nr_genes)).astype(int)
    population_per_level = np.array([len(master_genes)] + population_per_level.tolist())
    population_per_level[-1] += nr_genes - population_per_level.sum()
    chance_per_level = {level: chance_modifier(level, max_connection_chance) for level in range(nr_dep_levels)}

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
            parent_pool = levels_to_genes[this_level]
            parent_mask = np.random.binomial(1, chance_per_level[this_level], size=len(parent_pool)).astype(bool)
            parents[this_level] = parent_pool[parent_mask]
            nr_coeffs = len(parents[this_level])
            signs = alternating_signs(nr_coeffs)
            coeffs += [1.5 * np.random.rand(nr_coeffs) * signs]

        if coeffs:
            coeffs = np.concatenate(coeffs)

        assignment_dict[gene] = [reduce(lambda x, y: x + y.tolist(), parents.values(), []),
                                 LinearAssignment(noise_coeff, offset, *coeffs),
                                 NoiseGenerator("normal",
                                                loc=0, scale=np.random.rand() * 1)]

gene_tex_names = {name: r"$G_{" + str(i) + "}$" for name, i in zip(gene_names, range(nr_genes))}
cn = SCM(assignment_dict, variable_tex_names=gene_tex_names)
print(cn)
cn.plot(alpha=0.5)
sample = cn.sample(100000)
print(sample)
print(np.vstack([sample.max(), sample.min()]))
rng = np.random.default_rng()
sample_exp = pd.DataFrame(rng.poisson(np.exp(sample)), columns=sample.columns, dtype=int)
print(sample_exp)
print(np.vstack([sample_exp.max(), sample_exp.min()]))
