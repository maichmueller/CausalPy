import scm
import assignments as assigns
import noise_models as nm
import numpy as np
from numpy.random import PCG64, Generator
from collections import Counter

if __name__ == '__main__':
    nr_genes = 10000

    levels_of_dependency = 10

    perc_genes_per_level = np.array(
        sorted(
            Generator(PCG64()).binomial(1000, p=0.1, size=10000),
            key=lambda x: x)
    )[slice(0, 1000, 100)]
    print(perc_genes_per_level)
    perc_genes_per_level = perc_genes_per_level / perc_genes_per_level.sum()
    nr_genes_per_level = (perc_genes_per_level * nr_genes).astype(int)
    if nr_genes_per_level.sum() < nr_genes:
        selection = Counter(np.random.choice(levels_of_dependency,
                                             p=perc_genes_per_level,
                                             size=nr_genes - nr_genes_per_level.sum())
                            )
        for s, v in selection.items():
            nr_genes_per_level[s] += v



    p=3
