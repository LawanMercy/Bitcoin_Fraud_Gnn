"""
features.py
-----------
Computes structural, temporal, and neighbourhood features
for every node in the Elliptic Bitcoin transaction graph.

Usage
-----
from src.features import FeatureBuilder
fb = FeatureBuilder(G, df)
full_df = fb.build_all()
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional


class FeatureBuilder:
    """
    Builds a rich feature matrix by combining:
        1. Original Elliptic node features (166-dim)
        2. Structural centrality features
        3. Temporal burst / recency features
        4. Neighbourhood contagion features

    Parameters
    ----------
    G  : nx.DiGraph — the full transaction graph
    df : pd.DataFrame — output of graph_builder.build_dataframe()
    k  : int — number of pivot nodes for approximate betweenness (default 500)
    """

    def __init__(self, G: nx.DiGraph, df: pd.DataFrame, k: int = 500):
        self.G  = G
        self.df = df
        self.k  = k
        self._node_ids = df["txId"].tolist()

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_all(self) -> pd.DataFrame:
        """Run all feature families and return the merged DataFrame."""
        print("Building structural features…")
        struct = self._structural()
        print("Building temporal features…")
        temporal = self._temporal()
        print("Building neighbourhood features…")
        neigh = self._neighbourhood()

        orig_feats = [f"f{i}" for i in range(1, 166)]
        base_cols  = ["txId", "time_step", "class", "class_label"] + orig_feats

        full = (
            self.df[base_cols]
            .merge(struct,   on="txId", how="left")
            .merge(temporal.drop("time_step", axis=1), on="txId", how="left")
            .merge(neigh,    on="txId", how="left")
            .fillna(0)
        )
        print(f"✅ Feature matrix: {full.shape[0]:,} rows × {full.shape[1]} cols")
        return full

    def feature_names(self) -> list:
        """Return list of engineered feature column names (excluding metadata)."""
        struct   = list(self._structural().columns[1:])
        temporal = [c for c in self._temporal().columns if c not in ("txId","time_step")]
        neigh    = list(self._neighbourhood().columns[1:])
        orig     = [f"f{i}" for i in range(1, 166)]
        return orig + struct + temporal + neigh

    # ── Private methods ────────────────────────────────────────────────────────

    def _structural(self) -> pd.DataFrame:
        G, nodes = self.G, self._node_ids
        in_d  = dict(G.in_degree())
        out_d = dict(G.out_degree())

        print("  Computing betweenness…")
        btw = nx.betweenness_centrality(G, normalized=True, k=self.k)
        print("  Computing PageRank…")
        pr  = nx.pagerank(G, alpha=0.85, max_iter=300)
        print("  Computing HITS…")
        hubs, auths = nx.hits(G, max_iter=100, normalized=True)

        return pd.DataFrame({
            "txId"        : nodes,
            "in_degree"   : [in_d.get(n, 0)     for n in nodes],
            "out_degree"  : [out_d.get(n, 0)    for n in nodes],
            "total_degree": [in_d.get(n,0)+out_d.get(n,0) for n in nodes],
            "degree_ratio": [out_d.get(n,0)/(in_d.get(n,0)+1e-6) for n in nodes],
            "betweenness" : [btw.get(n, 0)       for n in nodes],
            "pagerank"    : [pr.get(n, 0)        for n in nodes],
            "hub_score"   : [hubs.get(n, 0)      for n in nodes],
            "auth_score"  : [auths.get(n, 0)     for n in nodes],
        })

    def _temporal(self) -> pd.DataFrame:
        df = self.df[["txId", "time_step"]].copy()

        ts_vol   = self.df.groupby("time_step").size().rename("ts_volume")
        ts_ill   = (self.df[self.df["class"] == 1]
                    .groupby("time_step").size().rename("ts_illicit_count"))

        ts_stats = (
            pd.DataFrame({"time_step": range(1, 50)})
            .merge(ts_vol.reset_index(),  on="time_step", how="left")
            .merge(ts_ill.reset_index(), on="time_step", how="left")
            .fillna(0)
        )
        ts_stats["ts_illicit_rate"] = (
            ts_stats["ts_illicit_count"] / (ts_stats["ts_volume"] + 1e-6)
        )

        df = df.merge(ts_stats[["time_step", "ts_volume", "ts_illicit_rate"]],
                      on="time_step", how="left")
        df["time_recency"] = (df["time_step"] - 1) / 48.0
        return df

    def _neighbourhood(self) -> pd.DataFrame:
        G         = self.G
        label_map = self.df.set_index("txId")["class"].to_dict()
        records   = []

        for node in self._node_ids:
            preds  = list(G.predecessors(node))
            succs  = list(G.successors(node))
            all_nb = preds + succs

            nb_lbls  = [label_map.get(n, 0) for n in all_nb]
            n_total  = len(all_nb) + 1e-6

            pred_lbls     = [label_map.get(n, 0) for n in preds]
            pred_ill_r    = sum(l == 1 for l in pred_lbls) / (len(preds) + 1e-6)

            records.append({
                "txId"              : node,
                "nb_illicit_ratio"  : sum(l == 1 for l in nb_lbls) / n_total,
                "nb_licit_ratio"    : sum(l == 2 for l in nb_lbls) / n_total,
                "nb_unknown_ratio"  : sum(l == 0 for l in nb_lbls) / n_total,
                "nb_total"          : len(all_nb),
                "pred_illicit_ratio": pred_ill_r,
            })

        return pd.DataFrame(records)
