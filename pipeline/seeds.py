"""Directed seed reachability within the observed four-hop graph."""

import networkx as nx
import pandas as pd


def compute_seed_reach(G: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    """Return gid, distinct upstream seed count, and minimum seed distance.

    Only directed paths of length <=4 count. A seed never counts itself,
    including through a cycle, but its distance is always zero. Unreachable
    nodes have a missing distance. Neither argument is modified.
    """
    if not G.is_directed() or G.is_multigraph():
        raise ValueError("Ожидается простой направленный граф")
    if not {"gid", "is_seed"}.issubset(nodes.columns):
        raise ValueError("Нужны колонки gid и is_seed")
    if nodes.gid.isna().any() or nodes.gid.duplicated().any():
        raise ValueError("gid должны быть заполнены и уникальны")
    if len(nodes) and (not pd.api.types.is_integer_dtype(nodes.gid.dtype)
                       or pd.api.types.is_bool_dtype(nodes.gid.dtype)
                       or nodes.gid.min() < -(2**63) or nodes.gid.max() > 2**63 - 1):
        raise ValueError("gid должны быть целыми числами в диапазоне int64; float запрещён")
    if nodes.is_seed.isna().any() or not nodes.is_seed.isin([True, False]).all():
        raise ValueError("is_seed должен содержать bool или 0/1")
    gids = nodes.gid.tolist()
    if set(G.nodes) != set(gids):
        raise ValueError("Граф должен содержать все gid, включая изоляты")
    counts = dict.fromkeys(gids, 0)
    distances = dict.fromkeys(gids, None)
    seeds = nodes.loc[nodes.is_seed.astype(bool), "gid"].tolist()
    for seed in seeds:
        distances[seed] = 0
        for gid, distance in nx.single_source_shortest_path_length(
            G, seed, cutoff=4
        ).items():
            if gid == seed:
                continue
            counts[gid] += 1
            if distances[gid] is None or distance < distances[gid]:
                distances[gid] = distance
    return pd.DataFrame({
        "gid": pd.Series(gids, dtype="int64"),
        "seed_reach_count": pd.Series([counts[g] for g in gids], dtype="int64"),
        "seed_distance": pd.array([distances[g] for g in gids], dtype="Int64"),
    })
