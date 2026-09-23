"""Направленные структурные метрики; seed-достижимость добавляет поток B."""

import networkx as nx
import numpy as np


def compute_features(G, nodes):
    """Return one row per input gid without mutating G or nodes.

    betweenness uses hop distance (weight=None), PageRank uses sum_kzt.
    Undefined ratios/average transfer amounts stay NaN.
    """
    if not G.is_directed() or G.is_multigraph():
        raise ValueError("Метрики требуют nx.DiGraph")
    if nodes["gid"].duplicated().any() or set(nodes["gid"]) != set(G):
        raise ValueError("Метрики: нужны уникальные gid и полный граф всех nodes")
    frame = nodes[["gid", "depth", "is_seed"]].copy().reset_index(drop=True)
    frame["gid"] = frame["gid"].astype("int64")
    for direction in ("in", "out"):
        degree = getattr(G, f"{direction}_degree")
        for suffix, weight, dtype in (("deg", None, "int64"),
                                       ("kzt", "sum_kzt", "float64"),
                                       ("tx", "n_tx", "int64")):
            frame[f"{direction}_{suffix}"] = frame["gid"].map(
                dict(degree(weight=weight))).astype(dtype)
        frame[f"{direction}_avg_kzt"] = (
            frame[f"{direction}_kzt"] / frame[f"{direction}_tx"].replace(0, np.nan)
        )
    pagerank = nx.pagerank(G, alpha=0.85, weight="sum_kzt", max_iter=1000, tol=1e-10)
    between = nx.betweenness_centrality(G, normalized=True, weight=None)
    frame["pagerank"] = frame["gid"].map(pagerank).astype(float)
    frame["betweenness"] = frame["gid"].map(between).astype(float)
    frame["pass_through"] = frame["out_kzt"] / frame["in_kzt"].replace(0, np.nan)
    frame["truncated_by_depth"] = frame["depth"].eq(4) & frame["out_deg"].eq(0)
    components = sorted(nx.weakly_connected_components(G), key=lambda c: (-len(c), min(c)))
    wcc = {gid: index for index, component in enumerate(components) for gid in component}
    frame["wcc_id"] = frame["gid"].map(wcc).astype("int64")
    return frame
