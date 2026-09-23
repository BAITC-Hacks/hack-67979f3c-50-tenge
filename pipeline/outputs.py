"""Validate and export the three analyst tables without mutating inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


NODE_COLUMNS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLUMNS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLUMNS = ["rank", "gid", "role", "priority_score", "why"]
ROLES = {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}


def _required(df, columns, name):
    if not df.columns.is_unique:
        raise ValueError(f"{name}: duplicate columns")
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if df[columns].isna().any().any():
        raise ValueError(f"{name}: null mandatory values")


def _integer(series, name):
    if not pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"{name}: expected integer dtype")
    if len(series) and (series.min() < -(2**63) or series.max() > 2**63 - 1):
        raise ValueError(f"{name}: outside int64 range")


def _numeric(series, name, lower=0, upper=None):
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"{name}: expected numeric dtype")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < lower).any() or (upper is not None and (values > upper).any()):
        raise ValueError(f"{name}: nonfinite or outside allowed range")


def _text(series, name):
    if not series.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{name}: expected nonempty text")


def write_outputs(scored_df, clusters_df, top_df, out_dir) -> None:
    """Write UTF-8 CSVs after checking schemas and cross-table consistency.

    Dataset coverage (2,248 gids) and sums against raw edges are checked by
    the caller, which has the raw inputs. Small graphs require min(20, N)
    top rows. Nullable extra metrics are preserved. Top can be non-seed only.
    """
    nodes, clusters, top = scored_df.copy(), clusters_df.copy(), top_df.copy()
    for df, cols, name in ((nodes, NODE_COLUMNS, "nodes"), (clusters, CLUSTER_COLUMNS, "clusters"), (top, TOP_COLUMNS, "top")):
        _required(df, cols, name)
    if nodes.empty:
        raise ValueError("nodes: empty graph")
    for df, cols in ((nodes, ["gid", "cluster_id"]), (clusters, ["cluster_id", "n_nodes", "n_seed"]), (top, ["rank", "gid"])):
        for col in cols:
            _integer(df[col], col)
    for df, col in ((nodes, "gid"), (clusters, "cluster_id"), (top, "gid")):
        if df[col].duplicated().any():
            raise ValueError(f"{col}: duplicate identifiers")
    for df, cols in ((nodes, ["role_score", "priority_score"]), (top, ["priority_score"])):
        for col in cols:
            _numeric(df[col], col, upper=1)
            df[col] = df[col].astype(float)
    for df in (nodes, top):
        if not df.role.isin(ROLES).all():
            raise ValueError("role: unknown role")
    for df, cols in ((nodes, ["evidence"]), (clusters, ["top_gids", "hypothesis"]), (top, ["why"])):
        for col in cols:
            _text(df[col], col)
    if nodes.evidence.str.len().gt(200).any() or not nodes.evidence.str.contains(r"\d", regex=True).all():
        raise ValueError("evidence: require a number and at most 200 characters")
    _numeric(clusters.sum_kzt_internal, "sum_kzt_internal")
    if (clusters.n_nodes <= 0).any() or (clusters.n_seed < 0).any() or (clusters.n_seed > clusters.n_nodes).any():
        raise ValueError("clusters: invalid node/seed counts")
    counts = nodes.groupby("cluster_id").size()
    summaries = clusters.set_index("cluster_id")
    if set(counts.index) != set(summaries.index) or not counts.eq(summaries.n_nodes.reindex(counts.index)).all():
        raise ValueError("clusters: membership/count mismatch")
    if "is_seed" in nodes:
        if nodes.is_seed.isna().any() or not nodes.is_seed.isin([True, False]).all():
            raise ValueError("is_seed: expected booleans")
        seed_counts = nodes.groupby("cluster_id").is_seed.sum()
        if not seed_counts.eq(summaries.n_seed.reindex(seed_counts.index)).all():
            raise ValueError("clusters: seed count mismatch")
    membership = nodes.set_index("gid").cluster_id.to_dict()
    stability_fields = {"cluster_membership_stability", "cluster_membership_status"}
    if stability_fields.intersection(nodes.columns):
        _required(nodes, sorted(stability_fields), "node stability")
        _numeric(nodes.cluster_membership_stability, "cluster_membership_stability", upper=1)
        if not nodes.cluster_membership_status.isin(["core", "disputed", "isolated"]).all():
            raise ValueError("node stability: unknown status")
        fields = {"stability_mean", "stability_min", "stability_status", "n_core", "n_disputed", "stability_runs"}
        _required(clusters, sorted(fields), "cluster stability")
        for field in ("stability_mean", "stability_min"):
            _numeric(clusters[field], field, upper=1)
        if (clusters.stability_min > clusters.stability_mean + 1e-12).any():
            raise ValueError("cluster stability: minimum exceeds mean")
        if not clusters.stability_status.isin(["stable", "variable", "unstable", "isolated"]).all():
            raise ValueError("cluster stability: unknown status")
        for field in ("n_core", "n_disputed", "stability_runs"):
            _integer(clusters[field], field)
            _numeric(clusters[field], field)
        if (clusters.stability_runs < 1).any():
            raise ValueError("cluster stability: no repeated runs")
        for status, field in (("core", "n_core"), ("disputed", "n_disputed")):
            actual = nodes.cluster_membership_status.eq(status).groupby(nodes.cluster_id).sum()
            if not clusters.set_index("cluster_id")[field].eq(actual).all():
                raise ValueError("cluster stability: membership count mismatch")
    for row in clusters.itertuples(index=False):
        try:
            gids = json.loads(row.top_gids)
        except (ValueError, TypeError) as exc:
            raise ValueError("top_gids: expected JSON array") from exc
        if not isinstance(gids, list) or not 1 <= len(gids) <= min(5, row.n_nodes) or any(type(gid) is not int for gid in gids):
            raise ValueError("top_gids: expected 1–5 integer gids")
        if len(set(gids)) != len(gids) or any(membership.get(gid) != row.cluster_id for gid in gids):
            raise ValueError("top_gids: duplicate or outside cluster")
    if not min(20, len(nodes)) <= len(top) <= len(nodes):
        raise ValueError("top: require at least min(20, node count) rows")
    if top['rank'].tolist() != list(range(1, len(top) + 1)):
        raise ValueError("top: ranks must be consecutive from 1")
    expected_order = top.sort_values(["priority_score", "gid"], ascending=[False, True]).gid.tolist()
    if top.gid.tolist() != expected_order:
        raise ValueError("top: order must be priority descending, gid ascending")
    indexed = nodes.set_index("gid")
    if not top.gid.isin(indexed.index).all():
        raise ValueError("top: unknown gid")
    selected = indexed.loc[top.gid]
    if selected.role.tolist() != top.role.tolist() or not np.allclose(selected.priority_score, top.priority_score, atol=1e-12, rtol=0):
        raise ValueError("top: role/score mismatch with nodes")
    contributions = ["priority_volume", "priority_bridge", "priority_seed", "priority_degree"]
    if any(col in nodes for col in contributions):
        if not all(col in nodes for col in contributions):
            raise ValueError("priority contributions: incomplete columns")
        for col in contributions:
            _numeric(nodes[col], col, upper=1)
        if not np.allclose(nodes[contributions].sum(axis=1), nodes.priority_score, atol=1e-12, rtol=0):
            raise ValueError("priority contributions: sum mismatch")
    if {"depth", "out_deg"}.issubset(nodes.columns):
        if ((nodes.depth == 4) & (nodes.out_deg == 0) & (nodes.role == "terminal")).any():
            raise ValueError("role: terminal forbidden at depth boundary")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for df, cols, sort_by, filename in (
        (nodes, NODE_COLUMNS, "gid", "nodes_roles.csv"),
        (clusters, CLUSTER_COLUMNS, "cluster_id", "clusters.csv"),
        (top, TOP_COLUMNS, "rank", "top_nodes.csv"),
    ):
        ordered = cols + [col for col in df.columns if col not in cols]
        df.sort_values(sort_by)[ordered].to_csv(output / filename, index=False, encoding="utf-8", lineterminator="\n")
