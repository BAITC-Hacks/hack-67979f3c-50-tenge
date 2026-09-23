"""Deterministic weighted communities; summaries retain directed cash flows."""

from __future__ import annotations

import json
import math

import networkx as nx
import pandas as pd


SUMMARY_COLUMNS = [
    "cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"
]
ROLE_LABELS = {
    "consolidator": "признаки консолидации",
    "transit": "признаки транзита",
    "distributor": "признаки распределения",
    "terminal": "наблюдаемые конечные получатели",
    "coordinator": "предполагаемые связующие узлы",
    "peripheral": "роль не определена",
}


def _validate_nodes(G: nx.DiGraph, nodes: pd.DataFrame) -> None:
    if not G.is_directed() or G.is_multigraph():
        raise ValueError("Ожидается направленный граф nx.DiGraph без кратных рёбер")
    if "gid" not in nodes:
        raise ValueError("В таблице отсутствует gid")
    if nodes["gid"].isna().any() or nodes["gid"].duplicated().any():
        raise ValueError("gid должны быть заполнены и уникальны")
    if set(nodes["gid"]) != set(G):
        raise ValueError("Множество gid таблицы и графа должно совпадать, включая изоляты")


def _undirected_projection(G: nx.DiGraph) -> nx.Graph:
    """Sum both directions instead of letting to_undirected overwrite a weight."""
    projection = nx.Graph()
    projection.add_nodes_from(sorted(G.nodes))
    for src, dst, attrs in sorted(G.edges(data=True), key=lambda edge: edge[:2]):
        try:
            amount = float(attrs["sum_kzt"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Каждое ребро должно содержать числовую sum_kzt") from exc
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("sum_kzt должна быть конечной положительной суммой")
        if projection.has_edge(src, dst):
            projection[src][dst]["sum_kzt"] += amount
        else:
            projection.add_edge(src, dst, sum_kzt=amount)
    return projection


def assign_clusters(
    G: nx.DiGraph, nodes: pd.DataFrame, *, resolution: float = 1.0, seed: int = 42
) -> pd.DataFrame:
    """Return gid/cluster_id, sorted by gid; inputs remain unchanged.

    Louvain runs separately within weak components on a weighted undirected
    projection. IDs start at zero: descending size, then ascending minimum gid.
    Resolution/seed are explicit for reproducibility and sensitivity checks.
    """
    _validate_nodes(G, nodes)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution должна быть конечной положительной величиной")
    projection = _undirected_projection(G)
    communities: list[set[int]] = []
    for component in sorted(nx.connected_components(projection), key=min):
        subgraph = projection.subgraph(sorted(component)).copy()
        if subgraph.number_of_edges() == 0:
            communities.extend({gid} for gid in sorted(component))
            continue
        found = nx.community.louvain_communities(
            subgraph, weight="sum_kzt", resolution=resolution, seed=seed
        )
        for community in found:
            # Defensive split: every returned cluster must be connected.
            communities.extend(set(part) for part in nx.connected_components(
                subgraph.subgraph(community)
            ))
    communities.sort(key=lambda members: (-len(members), min(members)))
    rows = [(gid, cluster_id) for cluster_id, members in enumerate(communities)
            for gid in sorted(members)]
    return pd.DataFrame(rows, columns=["gid", "cluster_id"]).astype(
        {"gid": "int64", "cluster_id": "int64"}
    ).sort_values("gid", ignore_index=True)


def summarize_clusters(G: nx.DiGraph, scored_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize clusters with original directed amounts and priority top five.

    Internal turnover counts each observed directed edge once, including both
    edges of a reciprocal pair. External incoming/outgoing flows refer only to
    other observed clusters, never to missing bank activity.
    """
    _validate_nodes(G, scored_df)
    required = {"cluster_id", "is_seed", "role", "priority_score"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки для сводки кластеров: {sorted(missing)}")
    if scored_df[list(required)].isna().any().any():
        raise ValueError("Обязательные поля сводки кластеров не должны быть пустыми")
    if not scored_df["is_seed"].isin([True, False, 0, 1]).all():
        raise ValueError("is_seed должен содержать логические значения или 0/1")
    if not scored_df["role"].isin(ROLE_LABELS).all():
        raise ValueError("Неизвестная роль в сводке кластеров")
    if not scored_df["priority_score"].map(
        lambda value: math.isfinite(value) and 0 <= value <= 1
    ).all():
        raise ValueError("priority_score должен быть конечным числом от 0 до 1")
    if not scored_df["cluster_id"].map(
        lambda value: math.isfinite(value) and value == int(value)
    ).all():
        raise ValueError("cluster_id должен быть целым числом")

    mapping = scored_df.set_index("gid")["cluster_id"].to_dict()
    flows = {cluster: [0.0, 0.0, 0.0] for cluster in mapping.values()}
    for src, dst, attrs in sorted(G.edges(data=True), key=lambda edge: edge[:2]):
        amount = float(attrs["sum_kzt"])
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("sum_kzt должна быть конечной положительной суммой")
        src_cluster, dst_cluster = mapping[src], mapping[dst]
        if src_cluster == dst_cluster:
            flows[src_cluster][0] += amount
        else:
            flows[src_cluster][2] += amount
            flows[dst_cluster][1] += amount

    rows = []
    for cluster_id, group in scored_df.groupby("cluster_id", sort=True):
        n_nodes = len(group)
        n_seed = int(group["is_seed"].sum())
        internal, incoming, outgoing = flows[cluster_id]
        top = group.sort_values(["priority_score", "gid"], ascending=[False, True])
        top_gids = json.dumps([int(gid) for gid in top["gid"].head(5)])
        dominant = sorted(group["role"].value_counts().items(), key=lambda item: (-item[1], item[0]))[:2]
        role_text = "; ".join(f"{ROLE_LABELS[role]}: {count}/{n_nodes}" for role, count in dominant)
        hypothesis = (
            f"Гипотеза по структуре переводов: {role_text}. "
            f"Seed: {n_seed}/{n_nodes} ({100*n_seed/n_nodes:.1f}%). "
            f"Внутри {internal:,.0f} KZT; с другими кластерами: "
            f"вход {incoming:,.0f}, выход {outgoing:,.0f} KZT. "
            "Связность не доказывает общую деятельность."
        )
        if n_nodes == 1 and G.degree(group["gid"].iloc[0]) == 0:
            hypothesis = (
                f"Изолят: 1 узел, seed {n_seed}; 0 наблюдаемых переводов. "
                "Для гипотезы о назначении недостаточно данных."
            )
        rows.append([int(cluster_id), n_nodes, n_seed, internal, top_gids, hypothesis])
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).astype({
        "cluster_id": "int64", "n_nodes": "int64", "n_seed": "int64",
        "sum_kzt_internal": "float64",
    })
