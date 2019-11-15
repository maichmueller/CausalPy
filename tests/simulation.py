from functools import reduce

from scm import *
import numpy as np
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from functools import partial

if __name__ == '__main__':

    # =====================
    # PARAMETERS
    # --------------------
    nr_genes = 15

    gene_names = [r"G_" + str(i) + "" for i in range(nr_genes)]
    gene_tex_names = {name: r"$G_{" + str(i) + "}$" for name, i in zip(gene_names, range(nr_genes))}

    levels_of_dependency = 5

    master_connectivity = 0.9
    connect_perc_max = 0.1
    connect_perc_min = 0.001
    connect_perc_incr = 0.005
    connect_perc_samelevel = 0.01
    max_connected_levels = 3

    # =====================

    mid = int(levels_of_dependency / 2)
    perc_genes_per_level = {mid: 1 / 2}
    dep_range = range(1, mid + 1) if levels_of_dependency % 2 == 1 else range(1, mid)
    for i in dep_range:
        perc_genes_per_level[mid - i] = ((1 / 2) ** (i + 1)) / 2
        perc_genes_per_level[mid + i] = ((1 / 2) ** (i + 1)) / 2
    else:
        for i in range(1, mid):
            perc_genes_per_level[mid - i] = ((1 / 2) ** (i + 1)) / 2
            perc_genes_per_level[mid + i] = ((1 / 2) ** (i + 1)) / 2
    rem_perc = 1 - sum(perc_genes_per_level.values())
    perc_genes_per_level[mid] += rem_perc

    nr_genes_per_level = {level: int(perc * nr_genes) for level, perc in perc_genes_per_level.items()}
    nr_genes_per_level[0] = 1
    if sum(nr_genes_per_level.values()) < nr_genes:
        selection = Counter(
            np.random.choice(
                levels_of_dependency,
                p=list(perc_genes_per_level.values()),
                size=nr_genes - sum(nr_genes_per_level.values())
            )
        )
        for s, v in selection.items():
            nr_genes_per_level[s] += v

    nr_genes_per_level = {level: genes for (level, genes) in sorted(nr_genes_per_level.items(), key=lambda x: x[0])}
    genes_to_level = dict()
    levels_to_genes: dict = defaultdict(list)
    curr_offset = 0
    for level, amount_genes in nr_genes_per_level.items():
        for name in gene_names[curr_offset:curr_offset + amount_genes]:
            genes_to_level[name] = level
            levels_to_genes[level].append(name)
        curr_offset += amount_genes

    assignment_dict = dict()
    gene_names_pbar = tqdm(gene_names)
    for gene in gene_names_pbar:
        gene_names_pbar.set_description(f"Processing {gene}")
        gene_level = genes_to_level[gene]
        if gene_level == 0:
            noise_coeff = 1
            offset = 0
            coeffs = []
            parents = {}
        else:
            connectivity = connect_perc_samelevel
            # the gene's neighbours on the current level
            parent_pool = levels_to_genes[gene_level]

            # the index of the current gene in the sorted(!!) list
            start_index = parent_pool.index(gene) + 1
            nr_neighbours = len(parent_pool)
            end_index_neighbours = start_index + max(2, int(connectivity * nr_neighbours))
            # take the next {connect_perc_samelevel} % of genes in the list of genes on this level. By doing so I can avoid
            # running into accidental cycle creations within my graph. Since parents (prev_level adjacent nodes) are chosen
            # by random, this doesn't hurt the randomness of the overall association, even if it happens deterministically
            # on the same dependency level.
            parents = {gene_level: parent_pool[start_index:end_index_neighbours]}

            # defining the coefficients for the linear function.
            # offset is a random number in [0, 20]
            # noise coefficient is always 1
            # coefficients for parents are chosen at random from [0, 2]
            offset = 0
            noise_coeff = 0.01
            nr_coeffs = len(parents[gene_level])
            signs = [-1, 1] * (nr_coeffs // 2) if nr_coeffs % 2 == 0 else ([-1, 1] * (nr_coeffs // 2 + 1))[:nr_coeffs]
            coeffs = [10. * np.random.rand(nr_coeffs) * np.array(signs)]

            for this_level in range(gene_level - 1, max(1, gene_level - max_connected_levels) - 1, - 1):
                parent_pool = levels_to_genes[this_level]
                parents[this_level] = np.random.choice(
                    parent_pool,
                    size=max(
                        1,
                        int(
                            min(
                                connect_perc_max,
                                connect_perc_min + connect_perc_incr * (gene_level - this_level)
                            ) * len(parent_pool)
                        )),
                    replace=False
                )
                nr_coeffs = len(parents[this_level])
                signs = [-1, 1] * (nr_coeffs // 2) if nr_coeffs % 2 == 0 else ([-1, 1] * (nr_coeffs // 2 + 1))[:nr_coeffs]
                coeffs += [.1 * np.random.rand(nr_coeffs) * np.array(signs)]

        assignment_dict[gene] = [reduce(lambda x, y: x + y.tolist(), parents.values(), initial=[]),
                                 LinearAssignment(noise_coeff, offset, *np.concatenate(coeffs)),
                                 NoiseGenerator("normal",
                                                loc=0, scale=np.random.rand() * 0.001)]

    cn = SCM(assignment_dict, variable_tex_names=gene_tex_names)
    # print(cn)
    cn.plot()
    sample = cn.sample(10000)
    print(sample)
    print(np.vstack([sample.max(), sample.min()]))
    rng = np.random.default_rng()
    sample_exp = rng.poisson(np.exp(sample))
    print(sample_exp)
    print(np.vstack([sample_exp.max(), sample_exp.min()]))
