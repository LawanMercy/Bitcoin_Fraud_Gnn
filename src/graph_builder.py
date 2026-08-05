"""
graph_builder.py
----------------
Utility functions for loading and constructing the Elliptic
Bitcoin transaction graph with full node/edge attributes.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

CLASS_MAP   = {1.0: "Illicit", 2.0: "Licit", 0.0: "Unknown"}
CLASS_COLOR = {"Illicit": "#FF4444", "Licit": "#00C9A7", "Unknown": "#AAAAAA"}


def load_raw(data_dir: Path = DATA_DIR):
    """
    Load the three Elliptic CSV files.

    Returns
    -------
    features : pd.DataFrame  — node features (txId, time_step, f1…f165)
    classes  : pd.DataFrame  — node labels   (txId, class)
    edges    : pd.DataFrame  — edge list      (txId1, txId2)
    """
    feat_cols = ["txId", "time_step"] + [f"f{i}" for i in range(1, 166)]
    features  = pd.read_csv(data_dir / "elliptic_txs_features.csv",
                            header=None, names=feat_cols)
    classes   = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    edges     = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    return features, classes, edges


def build_dataframe(features, classes) -> pd.DataFrame:
    """
    Merge features and labels into a single DataFrame.
    Converts string/numeric class labels to:
        0 = Unknown, 1 = Illicit, 2 = Licit
    """
    df = features.merge(classes, on="txId", how="left")
    df["class"] = (
        df["class"]
        .replace({"unknown": 0, "1": 1, "2": 2, 1: 1, 2: 2})
        .fillna(0)
        .astype(float)
    )
    df["class_label"] = df["class"].map(CLASS_MAP)
    return df


def build_graph(df: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """
    Construct a directed NetworkX graph with node attributes:
        - class       (0/1/2)
        - class_label (Unknown/Illicit/Licit)
        - time_step   (int 1–49)

    Parameters
    ----------
    df    : merged DataFrame from build_dataframe()
    edges : edge-list DataFrame

    Returns
    -------
    G : nx.DiGraph
    """
    G = nx.from_pandas_edgelist(
        edges, source="txId1", target="txId2",
        create_using=nx.DiGraph()
    )

    # Attach node attributes
    attr_cols = ["class", "class_label", "time_step"]
    for col in attr_cols:
        mapping = df.set_index("txId")[col].to_dict()
        nx.set_node_attributes(G, mapping, col)

    return G


def subgraph_by_class(G: nx.DiGraph, classes: list) -> nx.DiGraph:
    """
    Return the induced subgraph containing only nodes whose
    class_label is in `classes`.

    Example
    -------
    ill_sub = subgraph_by_class(G, ["Illicit"])
    """
    nodes = [n for n, d in G.nodes(data=True)
             if d.get("class_label") in classes]
    return G.subgraph(nodes).copy()


def summary(G: nx.DiGraph) -> dict:
    """Print and return a summary of graph properties."""
    wcc  = list(nx.weakly_connected_components(G))
    info = {
        "nodes"       : G.number_of_nodes(),
        "edges"       : G.number_of_edges(),
        "density"     : nx.density(G),
        "n_wcc"       : len(wcc),
        "largest_wcc" : max(len(c) for c in wcc) if wcc else 0,
    }
    for k, v in info.items():
        print(f"  {k:<15}: {v}")
    return info
