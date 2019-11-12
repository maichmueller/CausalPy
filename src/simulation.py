from functools import reduce

from src import noise_models as nm, scm, assignments as assigns
import numpy as np
from collections import Counter, defaultdict
from tqdm.auto import tqdm

if __name__ == '__main__':

    # =====================
    # PARAMETERS
    # --------------------
    nr_genes = 20000

    levels_of_dependency = 20

    connect_perc_max = 0.3
    connect_perc_min = 0.05
    connect_perc_incr = 0.1
    connect_perc_samelevel = 0.1
    max_connected_levels = 3

    # =====================

    mid = int(levels_of_dependency / 2)
    perc_genes_per_level = {mid: 1 / 2}
    if levels_of_dependency % 2 == 1:
        for i in range(1, mid + 1):
            perc_genes_per_level[mid - i] = ((1 / 2) ** (i + 1)) / 2
            perc_genes_per_level[mid + i] = ((1 / 2) ** (i + 1)) / 2
    else:
        perc_genes_per_level[0] = ((1 / 2) ** (mid + 1)) / 2
        for i in range(1, mid):
            perc_genes_per_level[mid - i] = ((1 / 2) ** (i + 1)) / 2
            perc_genes_per_level[mid + i] = ((1 / 2) ** (i + 1)) / 2
    rem_perc = 1 - sum(perc_genes_per_level.values())
    perc_genes_per_level[mid] += rem_perc

    nr_genes_per_level = {level: int(perc * nr_genes) for level, perc in perc_genes_per_level.items()}
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

    gene_names = [r"G_" + str(i) + "" for i in range(nr_genes)]
    gene_tex_names = {name: r"$G_{" + str(i) + "}$" for name, i in zip(gene_names, range(nr_genes))}
    nr_genes_per_level = {level: genes for (level, genes) in sorted(nr_genes_per_level.items(), key=lambda x: x[0])}
    genes_to_level = dict()
    levels_to_genes = defaultdict(list)
    curr_offset = 0
    for level, amount_genes in nr_genes_per_level.items():
        for name in gene_names[curr_offset:curr_offset + amount_genes]:
            genes_to_level[name] = level
            levels_to_genes[level].append(name)
        curr_offset += amount_genes

    assignment_dict = dict()
    for gene in (pbar := tqdm(gene_names)):
        pbar.set_description(f"Processing {gene}")
        gene_level = genes_to_level[gene]
        # the gene's neighbours on the current level
        parent_pool = levels_to_genes[gene_level]

        # the index of the current gene in the sorted(!!) list
        start_index = parent_pool.index(gene) + 1
        nr_neighbours = len(parent_pool)
        end_index_neighbours = start_index + max(2, int(connect_perc_samelevel * nr_neighbours))
        # take the next {connect_perc_samelevel} % of genes in the list of genes on this level. By doing so I can avoid
        # running into accidental cycle creations within my graph. Since parents (prev_level adjacent nodes) are chosen
        # by random, this doesn't hurt the randomness of the overall association, even if it happens deterministically
        # on the same dependency level.
        parents = {gene_level: parent_pool[start_index:end_index_neighbours]}

        # defining the coefficients for the linear function.
        # offset is a random number in [0, 20]
        # noise coefficient is always 1
        # coefficients for parents are chosen at random from [0, 2]
        offset = np.random.rand() * np.random.randint(0, 2)
        noise_coeff = 1
        nr_coeffs = len(parents[gene_level])
        coeffs = [np.random.rand(nr_coeffs) * np.random.choice([-1, 1], size=nr_coeffs)]
        # the final assignment function will be a linear one
        func = assigns.LinearAssignment
        for this_level in range(gene_level - 1, max(0, gene_level - max_connected_levels) - 1, - 1):
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
            coeffs += [np.random.rand(nr_coeffs) * np.random.choice([-1, 1], size=nr_coeffs)]

        assignment_dict[gene] = [reduce(lambda x, y: x + y.tolist(), parents.values()),
                                 func(noise_coeff, offset, *np.concatenate(coeffs)),
                                 nm.NoiseGenerator("normal",
                                                   loc=np.random.rand() * 20, scale=np.random.rand() * 5)]

    cn = scm.SCM(assignment_dict, variable_tex_names=gene_tex_names)
    print(cn)
    cn.plot()
    sample = cn.sample(100000)

    print(sample)
    print(np.vstack([sample.max(), sample.min()]))
