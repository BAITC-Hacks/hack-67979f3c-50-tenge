"""One-command graph analysis pipeline."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pandas as pd


def _merge_nodes(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    if "gid" not in left or "gid" not in right:
        raise ValueError(f"{label}: missing gid")
    if left.gid.isna().any() or right.gid.isna().any():
        raise ValueError(f"{label}: null gid")
    if not left.gid.is_unique or not right.gid.is_unique:
        raise ValueError(f"{label}: duplicate gid")
    if set(left.gid) != set(right.gid):
        raise ValueError(f"{label}: gid sets differ")
    overlap = (set(left.columns) & set(right.columns)) - {"gid"}
    if overlap:
        raise ValueError(f"{label}: overlapping columns: {sorted(overlap)}")
    return left.merge(right, on="gid", how="left", validate="one_to_one", sort=False)


def main(data_dir: Path, out_dir: Path) -> None:
    try:
        from pipeline.io import load_data, validate_inputs
        from pipeline.graph import build_graph
        from pipeline.features import compute_features
        from pipeline.roles import assign_roles
        from pipeline.seeds import compute_seed_reach
        from pipeline.clusters import assign_clusters, summarize_clusters
        from pipeline.stability import analyze_stability
        from pipeline.temporal import analyze_temporal
        from pipeline.routes import analyze_routes
        from pipeline.priority import compute_priority, make_top_nodes
        from pipeline.outputs import write_outputs
    except ImportError as exc:
        raise SystemExit(f"Модули потоков A/B пока не готовы: {exc}") from exc

    started = perf_counter()
    nodes, edges, transactions = load_data(data_dir)
    validate_inputs(nodes, edges, transactions)
    graph = build_graph(nodes, edges)
    if set(graph.nodes) != set(nodes.gid):
        raise ValueError("Граф потерял узлы из nodes.parquet, включая изоляты")

    features = compute_features(graph, nodes)
    seeds = compute_seed_reach(graph, nodes)
    enriched = _merge_nodes(features, seeds, "seed metrics")
    roles = assign_roles(enriched)
    thresholds = roles.attrs.get("role_thresholds", {})
    candidate_counts = roles.attrs.get("role_candidate_counts", {})
    clusters = assign_clusters(graph, nodes)
    stability_nodes, stability_clusters, stability_diagnostics = analyze_stability(graph, nodes, clusters)
    with_clusters = _merge_nodes(roles, clusters, "clusters")
    with_clusters = _merge_nodes(with_clusters, stability_nodes, "cluster stability")
    scored = compute_priority(with_clusters)
    from pipeline.priority_sensitivity import analyze_priority_sensitivity
    priority_sensitivity = analyze_priority_sensitivity(scored)
    temporal_nodes, temporal_report = analyze_temporal(nodes, transactions, scored)
    scored = _merge_nodes(scored, temporal_nodes, "temporal patterns")
    network_report = analyze_routes(graph, transactions, scored)
    if not scored.gid.is_unique or set(scored.gid) != set(nodes.gid):
        raise ValueError("Приоритет потерял или продублировал gid")
    cluster_summary = summarize_clusters(graph, scored)
    if (not stability_clusters.cluster_id.is_unique
            or set(stability_clusters.cluster_id) != set(cluster_summary.cluster_id)):
        raise ValueError("Устойчивость: набор кластеров не совпадает со сводкой")
    cluster_summary = cluster_summary.merge(stability_clusters, on="cluster_id", validate="one_to_one")
    top = make_top_nodes(scored, limit=30)
    write_outputs(scored, cluster_summary, top, out_dir)
    for filename, report in (("temporal_patterns.json", temporal_report),
                             ("network_patterns.json", network_report)):
        (out_dir / filename).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metadata = {
        "role_thresholds": thresholds,
        "role_candidate_counts": candidate_counts,
        "cluster_seed": 42,
        "cluster_stability": stability_diagnostics,
        "priority_sensitivity": priority_sensitivity,
        "bonus_reports": ["temporal_patterns.json", "network_patterns.json"],
        "nodes": len(nodes), "edges": len(edges), "transactions": len(transactions),
        "duration_seconds": round(perf_counter() - started, 3),
        "versions": {"python": sys.version.split()[0], **{
            package: version(package) for package in
            ("pandas", "numpy", "networkx", "pyarrow", "scipy")}},
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Готово: {len(scored)} узлов, {len(cluster_summary)} кластеров, "
          f"{len(top)} в top; {perf_counter() - started:.2f} с")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ графа переводов")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    main(args.data, args.out)
