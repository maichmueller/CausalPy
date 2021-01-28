import pandas as pd
import networkx as nx
import os


def load_cropseq_small():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(os.path.join(dir_path, "data/CROPseq_data_small.csv")).drop(
        columns=["Unnamed: 0", "cell.barcode"]
    )
    gene_names = pd.unique(data["gene_targeted"])
    environments = data.gene_targeted.astype("category").cat.codes
    data.drop(columns="gene_targeted", inplace=True)

    gt_graph = nx.DiGraph()
    gt_graph.add_edge("LCK", "ZAP70", label="+")
    gt_graph.add_edge("ZAP70", "LAT", label="+")
    gt_graph.add_edge("PTPN6", "LCK", label="-")
    gt_graph.add_edge("DOK2", "LCK", label="-")
    gt_graph.add_edge("DOK2", "ZAP70", label="-")
    gt_graph.add_node("PTPN11")
    gt_graph.add_node("EGR3")
    return data, environments, gene_names, gt_graph
