import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional, List, Collection

from causalpy import Assignment
from causalpy.neural_networks import FCNet
from causalpy.causal_prediction.interventional import (
    AgnosticPredictor,
    MultiAgnosticPredictor,
)
from examples.study_cases import study_scm, generate_data_from_scm
import numpy as np
import torch
from plotly import graph_objs as go
from time import gmtime, strftime
import os

from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use("science")
import matplotlib.patches as mpatches
import seaborn as sns
from random_graphs import random_graphs
import networkx as nx

# for i in range(100):
#     scm, a, b = random_graphs(
#         15, device="cuda", p_con=1, p_con_down=lambda x, y: 1, seed=i
#     )
#     try:
#         scm.sample(10)
#     except:
#         x = 3
# #         scm, a, b = random_graphs(
# #             15, device="cuda", p_con=1, p_con_down=lambda x, y: 1, seed=i
# #         )
# #         t = 3
# for var in scm.get_variables():
#     print(scm[var][1]["assignment"].dim_in, len(scm[var][0]))

# scm = random_graphs(15, device="cuda", p_con=0.5, p_con_down=lambda x, y: 1, seed=2)
scm = random_graphs(15, device="cuda", p_con=0.4, seed=2)
scm.plot(alpha=1)
plt.show()
sample = scm.sample(10000)
sample.columns = [s.replace("_", "") for s in sample.columns]
sample.hist(figsize=(10, 8), bins=150)
plt.show()
