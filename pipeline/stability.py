"""Seed sensitivity of observed communities and an experimental sparse consensus.

No baseline assignment is replaced. Stability measures algorithmic agreement,
not evidence of common activity or the correctness of a financial hypothesis.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import math
from time import perf_counter

import networkx as nx
import pandas as pd

from pipeline.clusters import _undirected_projection, _validate_nodes, assign_clusters


ALTERNATE_SEEDS = (0, 1, 7, 21, 99, 101, 202, 303, 404, 505)
NODE_COLUMNS = ["gid", "cluster_membership_stability", "cluster_membership_status"]
CLUSTER_COLUMNS = [
    "cluster_id", "stability_mean", "stability_min", "stability_status",
    "n_core", "n_disputed", "stability_runs",
]


def _mapping(frame, gids):
    if not frame.columns.is_unique or not {"gid", "cluster_id"}.issubset(frame.columns):
        raise ValueError("Нужны уникальные колонки gid и cluster_id")
    for column in ("gid", "cluster_id"):
        values = frame[column]
        if (values.isna().any() or not pd.api.types.is_integer_dtype(values.dtype)
                or pd.api.types.is_bool_dtype(values.dtype)):
            raise ValueError(f"{column}: требуется целочисленный тип без пропусков")
        if len(values) and (values.min() < -(2**63) or values.max() > 2**63 - 1):
            raise ValueError(f"{column}: вне диапазона int64")
    if frame.gid.duplicated().any() or set(frame.gid) != set(gids):
        raise ValueError("Карта кластеров должна содержать каждый gid ровно один раз")
    # Never use iterrows or a mixed numeric ndarray: gid can exceed 2**53.
    return {int(gid): int(cluster) for gid, cluster in zip(frame.gid, frame.cluster_id)}


def _groups(mapping):
    groups = defaultdict(set)
    for gid, cluster in mapping.items():
        groups[cluster].add(gid)
    return dict(groups)


def _adjusted_rand(left, right):
    """Exact integer contingency arithmetic, without a dense node-pair matrix."""
    n = len(left)
    if n < 2:
        return 1.0
    cells = Counter((left[gid], right[gid]) for gid in left)
    rows, columns = Counter(left.values()), Counter(right.values())

    def pairs(counts):
        return sum(count * (count - 1) // 2 for count in counts)

    common = pairs(cells.values())
    a, b = pairs(rows.values()), pairs(columns.values())
    total = n * (n - 1) // 2
    numerator = 2 * (common * total - a * b)
    denominator = (a + b) * total - 2 * a * b
    return numerator / denominator if denominator else 1.0


def _partition_metrics(projection, mapping, ensemble, alternatives):
    groups = list(_groups(mapping).values())
    total = math.fsum(data["sum_kzt"] for _, _, data in projection.edges(data=True))
    internal = math.fsum(
        data["sum_kzt"] for src, dst, data in projection.edges(data=True)
        if mapping[src] == mapping[dst]
    )
    return {
        "n_clusters": len(groups),
        "modularity": (float(nx.community.modularity(
            projection, groups, weight="sum_kzt", resolution=1.0
        )) if total > 0 else None),
        "amount_retention": internal / total if total > 0 else None,
        "sum_kzt_internal": internal,
        "mean_ari_to_ensemble": math.fsum(
            _adjusted_rand(mapping, member) for member in ensemble
        ) / len(ensemble),
        "mean_ari_to_alternatives": math.fsum(
            _adjusted_rand(mapping, member) for member in alternatives
        ) / len(alternatives),
    }


def analyze_stability(G, nodes, baseline_map_df):
    """Return (node metrics, cluster metrics, JSON-compatible diagnostics).

    Ten alternative seeds run the production clustering at resolution 1.
    For each baseline cluster and each run, choose the alternative group
    maximizing Jaccard overlap, resolving ties by minimum member gid.
    A node's score is the fraction of runs where it belongs to that group.
    Cluster scores are mean/minimum best Jaccard. Each matching is independent,
    so several baseline groups may choose the same alternative after a merge.

    The experimental consensus reweights only existing undirected edges by
    coassignment frequency across baseline plus ten runs. Zero-frequency edges
    are removed; all vertices remain. It is evaluated on the original graph,
    and its partition is never returned as the production cluster assignment.
    Inputs are not modified. Storage is O(runs*V + E), without V-by-V matrices.
    """
    started = perf_counter()
    _validate_nodes(G, nodes)
    gids = sorted(int(gid) for gid in nodes.gid)
    baseline = _mapping(baseline_map_df, gids)
    baseline_groups = _groups(baseline)
    projection = _undirected_projection(G)
    isolated = set(nx.isolates(projection))
    if any(len(baseline_groups[baseline[gid]]) != 1 for gid in isolated):
        raise ValueError("Изолят должен быть отдельным базовым кластером")

    alternative_started = perf_counter()
    alternatives = [
        _mapping(assign_clusters(G, nodes, resolution=1.0, seed=seed), gids)
        for seed in ALTERNATE_SEEDS
    ]
    alternative_seconds = perf_counter() - alternative_started
    overlaps = {cluster: [] for cluster in baseline_groups}
    memberships = dict.fromkeys(gids, 0)
    for alternative in alternatives:
        alternative_groups = _groups(alternative)
        sizes = {cluster: len(group) for cluster, group in alternative_groups.items()}
        minima = {cluster: min(group) for cluster, group in alternative_groups.items()}
        contingency = defaultdict(Counter)
        for gid in gids:
            contingency[baseline[gid]][alternative[gid]] += 1
        for cluster, group in baseline_groups.items():
            matches = contingency[cluster]

            def jaccard(other):
                intersection = matches[other]
                return Fraction(intersection, len(group) + sizes[other] - intersection)

            best = min(matches, key=lambda other: (-jaccard(other), minima[other]))
            overlaps[cluster].append(float(jaccard(best)))
            for gid in group:
                memberships[gid] += alternative[gid] == best

    runs = len(ALTERNATE_SEEDS)
    node_rows = []
    statuses = {}
    for gid in gids:
        score = memberships[gid] / runs
        status = "isolated" if gid in isolated else "core" if score >= 0.8 else "disputed"
        statuses[gid] = status
        node_rows.append((gid, 1.0 if gid in isolated else score, status))
    node_metrics = pd.DataFrame(node_rows, columns=NODE_COLUMNS).astype({
        "gid": "int64", "cluster_membership_stability": "float64",
        "cluster_membership_status": "str",
    })
    cluster_rows = []
    for cluster, group in sorted(baseline_groups.items()):
        values = overlaps[cluster]
        minimum = min(values)
        is_isolate = len(group) == 1 and next(iter(group)) in isolated
        status = ("isolated" if is_isolate else "stable" if minimum >= 0.8
                  else "variable" if minimum >= 0.5 else "unstable")
        cluster_rows.append((
            cluster, math.fsum(values) / runs, minimum, status,
            sum(statuses[gid] == "core" for gid in group),
            sum(statuses[gid] == "disputed" for gid in group), runs,
        ))
    cluster_metrics = pd.DataFrame(cluster_rows, columns=CLUSTER_COLUMNS).astype({
        "cluster_id": "int64", "stability_mean": "float64", "stability_min": "float64",
        "stability_status": "str", "n_core": "int64", "n_disputed": "int64",
        "stability_runs": "int64",
    })

    consensus_started = perf_counter()
    ensemble = [baseline, *alternatives]
    consensus_graph = nx.DiGraph()
    consensus_graph.add_nodes_from(gids)
    for src, dst, data in projection.edges(data=True):
        fraction = sum(member[src] == member[dst] for member in ensemble) / len(ensemble)
        if fraction > 0:
            consensus_graph.add_edge(src, dst, sum_kzt=data["sum_kzt"] * fraction)
    # A single directed edge represents each undirected pair. assign_clusters
    # reconstructs the same summed projection and reuses component handling.
    consensus = _mapping(assign_clusters(
        consensus_graph, nodes, resolution=1.0, seed=42
    ), gids)
    consensus_seconds = perf_counter() - consensus_started
    comparison = {
        "baseline": _partition_metrics(projection, baseline, ensemble, alternatives),
        "consensus": _partition_metrics(projection, consensus, ensemble, alternatives),
    }
    original_weight = math.fsum(data["sum_kzt"] for _, _, data in projection.edges(data=True))
    consensus_weight = math.fsum(data["sum_kzt"] for _, _, data in consensus_graph.edges(data=True))
    diagnostics = {
        "settings": {
            "alternate_seeds": list(ALTERNATE_SEEDS), "alternate_runs": runs,
            "resolution": 1.0, "weight": "sum_kzt", "consensus_seed": 42,
            "consensus_ensemble_size": len(ensemble),
            "baseline_included_in_consensus": True,
            "core_threshold": 0.8, "stable_min_jaccard": 0.8,
            "variable_min_jaccard": 0.5,
            "matching": "max_jaccard_then_min_gid", "baseline_replaced": False,
        },
        "comparison": comparison,
        "n_nodes": len(gids), "n_isolates": len(isolated),
        "node_status_counts": dict(Counter(statuses.values())),
        "cluster_status_counts": dict(Counter(cluster_metrics.stability_status)),
        "consensus_edges": consensus_graph.number_of_edges(),
        "original_undirected_edges": projection.number_of_edges(),
        "consensus_reweighted_amount_fraction": (
            consensus_weight / original_weight if original_weight > 0 else None
        ),
        "metric_definitions": {
            "modularity": "Модулярность на исходной неориентированной проекции, resolution=1.",
            "amount_retention": "Доля исходной суммы переводов внутри групп; сумма каждой направленной связи учитывается один раз.",
            "mean_ari_to_ensemble": "Средний ARI к 11 разбиениям: базовому и 10 альтернативным; базовое сравнение с собой включено.",
            "mean_ari_to_alternatives": "Средний ARI к 10 альтернативным разбиениям; одинаковый набор сравнения для обоих методов.",
        },
        "limitations": [
            "Устойчивость к seed алгоритма не доказывает корректность групп или общую деятельность клиентов.",
            "Проверены 10 начальных состояний при фиксированных данных и resolution=1; неполнота данных не моделируется.",
            "Изоляты получают 1 технически; это не уверенность в роли или связях.",
            "При слиянии групп несколько базовых кластеров могут выбрать одну альтернативную группу.",
            "Узел может оставаться core при расширении его группы: учитывайте также Jaccard кластера.",
            "Разреженный консенсус учитывает только существующие связи и не добавляет рёбра между несвязанными парами.",
            "Согласие консенсуса с ансамблем измеряется на том же ансамбле, который его построил; это не независимая оценка качества.",
            "Модулярность оценивается на целом исходном графе; Louvain запускается отдельно в каждой компоненте.",
        ],
        "timing_seconds": {
            "alternate_runs": alternative_seconds, "consensus": consensus_seconds,
            "total": perf_counter() - started,
        },
    }
    return node_metrics, cluster_metrics, diagnostics
