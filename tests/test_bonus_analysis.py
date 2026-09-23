"""Independent small-graph checks for temporal patterns, routes and stability."""

import copy
import json

import networkx as nx
import pandas as pd
import pytest

from pipeline.clusters import assign_clusters
from pipeline.routes import analyze_routes
from pipeline.stability import analyze_stability
from pipeline.temporal import analyze_temporal


BASE = 2**53 + 1
A, B, C, D, ISOLATE = [BASE + index for index in range(5)]


def fixture(records, gids=(A, B, C, D, ISOLATE), depths=None):
    nodes = pd.DataFrame({"gid": list(gids), "depth": depths or [1] * len(gids),
                          "is_seed": [False] * len(gids)})
    tx = pd.DataFrame(records, columns=["src", "dst", "date", "sum_kzt"])
    graph = nx.DiGraph()
    graph.add_nodes_from(gids)
    for src, dst, _, amount in records:
        if graph.has_edge(src, dst):
            graph[src][dst]["sum_kzt"] += amount
            graph[src][dst]["n_tx"] += 1
        else:
            graph.add_edge(src, dst, sum_kzt=float(amount), n_tx=1)
    scored = pd.DataFrame({
        "gid": list(gids),
        "in_kzt": [graph.in_degree(gid, weight="sum_kzt") for gid in gids],
        "out_kzt": [graph.out_degree(gid, weight="sum_kzt") for gid in gids],
        "in_deg": [graph.in_degree(gid) for gid in gids],
        "out_deg": [graph.out_degree(gid) for gid in gids],
        "priority_score": [1 - index / max(1, len(gids)) for index in range(len(gids))],
    })
    return graph, nodes, tx, scored


def test_temporal_calendar_order_same_day_and_long_ids_without_mutation():
    _, nodes, tx, scored = fixture([
        (A, B, "2026-07-01T23:59:00", 5000),
        (A, B, "2026-07-02T23:59:00", 6000),
        (B, C, "2026-07-02T00:01:00", 7000),
        (B, C, "2026-07-04T00:01:00", 8000),
        (B, C, "2026-06-30T23:59:00", 9000),
    ])
    originals = [frame.copy(deep=True) for frame in (nodes, tx, scored)]
    metrics, report = analyze_temporal(nodes, tx, scored)
    detail = report["nodes"][str(B)]
    assert [(item["incoming_date"], item["outgoing_date"], item["lag_days"])
            for item in detail["lag_1_2_day_pairs"]] == [
        ("2026-07-01", "2026-07-02", 1), ("2026-07-02", "2026-07-04", 2),
    ]
    assert detail["same_day_flow_dates"] == ["2026-07-02"]
    indexed = metrics.set_index("gid")
    assert indexed.loc[B, "temporal_lag_1_2_day_pairs"] == 2
    assert indexed.loc[ISOLATE, "temporal_active_days"] == 0
    assert report["nodes"][str(ISOLATE)]["daily"] == []
    assert metrics.gid.dtype == "int64" and set(metrics.gid) == {A, B, C, D, ISOLATE}
    assert set(report["nodes"]) == {str(gid) for gid in nodes.gid}
    json.dumps(report, allow_nan=False)
    for actual, original in zip((nodes, tx, scored), originals):
        pd.testing.assert_frame_equal(actual, original)


def test_self_transfer_counts_once_and_does_not_supply_external_sender():
    _, nodes, tx, scored = fixture([
        (A, B, "2026-07-01", 5000), (C, B, "2026-07-01", 5000),
        (B, B, "2026-07-01", 5000),
        (A, B, "2026-07-02", 5000), (C, B, "2026-07-02", 5000),
        (D, B, "2026-07-02", 5000), (B, B, "2026-07-02", 5000),
    ])
    metrics, report = analyze_temporal(nodes, tx, scored)
    detail = report["nodes"][str(B)]
    assert detail["daily"][0]["in_tx"] == 3
    assert detail["daily"][0]["out_tx"] == 1
    assert detail["daily"][0]["activity_tx"] == 3
    assert detail["daily"][0]["activity_kzt"] == 15000
    assert metrics.set_index("gid").loc[B, "temporal_synchronous_days"] == 1
    assert detail["synchronous_incoming"][0]["date"] == "2026-07-02"
    assert detail["synchronous_incoming"][0]["sender_gids"] == [str(A), str(C), str(D)]


def test_equal_amount_groups_and_bursts_have_independent_count_and_money_evidence():
    _, nodes, tx, scored = fixture(
        [(A, B, "2026-07-01", 5000)] * 3
        + [(A, B, "2026-07-02", 5000), (A, B, "2026-07-03", 5000)]
    )
    metrics, report = analyze_temporal(nodes, tx, scored)
    assert report["equal_amount_groups_total"] == 1
    detail = report["nodes"][str(B)]
    assert detail["equal_amount_groups"] == [{
        "src": str(A), "dst": str(B), "date": "2026-07-01",
        "amount_kzt": 5000.0, "n_tx": 3, "sum_kzt": 15000.0,
    }]
    assert len(detail["bursts"]) == 1
    burst = detail["bursts"][0]
    assert burst["count_burst"] and burst["amount_burst"]
    assert burst["median_active_day_tx"] == 1
    assert burst["median_active_day_kzt"] == 5000
    assert metrics.set_index("gid").loc[B, "temporal_burst_days"] == 1


@pytest.mark.parametrize("cohort_size,expected", [(19, "insufficient_cohort"), (20, "outlier")])
def test_depth_anomaly_requires_adequate_cohort_and_explicit_upper_rank(cohort_size, expected):
    recipients = list(range(BASE + 100, BASE + 100 + cohort_size))
    records = [(A, gid, "2026-07-01", 100000 if gid == recipients[-1] else 5000)
               for gid in recipients]
    _, nodes, tx, scored = fixture(records, gids=[A, *recipients], depths=[0] + [1] * cohort_size)
    metrics, report = analyze_temporal(nodes, tx, scored)
    item = metrics.set_index("gid").loc[recipients[-1]]
    assert item.anomaly_status == expected
    assert item.anomaly_volume_percentile == 1
    assert item.anomaly_depth_cohort_size == cohort_size
    assert report["nodes"][str(recipients[-1])]["depth_anomaly"]["high_volume"] == (cohort_size == 20)


def test_temporal_missing_date_rejected():
    _, nodes, tx, scored = fixture([(A, B, None, 5000)])
    with pytest.raises(ValueError, match="пропуски"):
        analyze_temporal(nodes, tx, scored)


def test_repeated_route_requires_two_input_dates_and_preserves_leg_amounts():
    graph, _, tx, scored = fixture([
        (A, B, "2026-07-01", 5000), (A, B, "2026-07-02", 6000),
        (B, C, "2026-07-03", 7000), (B, C, "2026-07-03", 8000),
    ])
    report = analyze_routes(graph, tx, scored)
    routes = report["recurring_routes"]
    assert routes["eligible_total"] == 1
    route = routes["items"][0]
    assert route["gids"] == [str(A), str(B), str(C)]
    assert route["matched_incoming_days"] == 2 and route["matched_day_pairs"] == 2
    assert route["first_leg_sum_kzt"] == 11000
    assert route["second_leg_sum_kzt"] == 15000
    # The same outgoing day may support two day-pairs: it must not be summed twice as the leg total.
    assert sum(item["outgoing_sum_kzt"] for item in route["evidence"]) == 30000
    collapsed = tx.copy()
    collapsed.loc[collapsed.src.eq(A), "date"] = "2026-07-02"
    assert analyze_routes(graph, collapsed, scored)["recurring_routes"]["eligible_total"] == 0
    json.dumps(report, allow_nan=False)


def test_directed_cycles_have_unique_rotations_and_keep_opposite_directions():
    records = [(src, dst, "2026-07-01", 5000)
               for src in (A, B, C) for dst in (A, B, C) if src != dst]
    graph, _, tx, scored = fixture(records)
    cycles = analyze_routes(graph, tx, scored)["short_cycles"]
    assert cycles["by_length"] == {"2": 3, "3": 2}
    assert cycles["eligible_total"] == 5
    ids = [tuple(item["gids"]) for item in cycles["items"]]
    assert len(ids) == len(set(ids))
    assert (str(A), str(B), str(C)) in ids and (str(A), str(C), str(B)) in ids
    for item in cycles["items"]:
        assert item["sum_kzt_edges"] == 5000 * item["length"]


def test_resilience_same_n_and_original_weight_denominator():
    graph, _, tx, scored = fixture([
        (A, B, "2026-07-01", 5000), (B, C, "2026-07-02", 10000),
        (C, D, "2026-07-03", 15000),
    ])
    # Choose a bridge, so deleting one vertex also fragments surviving nodes.
    scored["priority_score"] = [0.5, 1.0, 0.4, 0.3, 0.0]
    report = analyze_routes(graph, tx, scored)["resilience"]
    first = report["scenarios"][0]
    assert first["removed_gids"] == [str(B)]
    assert first["remaining_node_count"] == 4
    assert first["largest_component_nodes"] == 2
    assert first["largest_component_fraction_remaining"] == 0.5
    assert first["retained_edge_weight_fraction"] == 0.5  # 15000 / original 30000.
    assert first["original_largest_remaining_nodes"] == 3
    assert first["original_largest_fragment_count"] == 2
    assert first["original_largest_nodes_outside_largest_fragment"] == 1
    for scenario in report["scenarios"]:
        assert scenario["remaining_node_count"] == len(graph) - scenario["removed_node_count"]
        assert scenario["random_baseline"]["largest_component_nodes_max"] <= scenario["remaining_node_count"]
        assert scenario["random_baseline"]["runs"] == 20


def test_stability_connected_communities_isolate_and_repeatability():
    graph, nodes, _, _ = fixture([
        (A, B, "2026-07-01", 10000), (B, A, "2026-07-01", 5000),
        (C, D, "2026-07-01", 12000),
    ])
    baseline = assign_clusters(graph, nodes)
    graph_original, nodes_original, baseline_original = copy.deepcopy(graph), nodes.copy(), baseline.copy()
    metrics, clusters, diagnostics = analyze_stability(graph, nodes, baseline)
    indexed = metrics.set_index("gid")
    assert indexed.loc[ISOLATE, "cluster_membership_status"] == "isolated"
    assert indexed.loc[ISOLATE, "cluster_membership_stability"] == 1
    assert metrics.cluster_membership_stability.eq(1).all()
    isolate_cluster = int(baseline.loc[baseline.gid.eq(ISOLATE), "cluster_id"].iloc[0])
    isolate_metrics = clusters.set_index("cluster_id").loc[isolate_cluster]
    assert isolate_metrics.stability_status == "isolated"
    assert isolate_metrics.n_core == isolate_metrics.n_disputed == 0
    assert clusters.stability_runs.eq(10).all()
    assert clusters.stability_min.eq(1).all()
    for group in baseline.groupby("cluster_id"):
        assert nx.is_connected(graph.subgraph(group[1].gid).to_undirected())
    for method in ("baseline", "consensus"):
        summary = diagnostics["comparison"][method]
        assert summary["n_clusters"] == 3
        assert summary["amount_retention"] == 1
        assert summary["mean_ari_to_ensemble"] == summary["mean_ari_to_alternatives"] == 1
    assert not diagnostics["settings"]["baseline_replaced"]
    again = analyze_stability(graph, nodes.iloc[::-1], baseline.iloc[::-1])
    pd.testing.assert_frame_equal(metrics, again[0])
    pd.testing.assert_frame_equal(clusters, again[1])
    assert diagnostics["comparison"] == again[2]["comparison"]
    assert nx.utils.graphs_equal(graph, graph_original)
    pd.testing.assert_frame_equal(nodes, nodes_original)
    pd.testing.assert_frame_equal(baseline, baseline_original)
    assert metrics.gid.dtype == "int64" and set(metrics.gid) == {A, B, C, D, ISOLATE}
    json.dumps(diagnostics, allow_nan=False)


def test_daily_svg_zero_and_single_day():
    from viz.bonus_panels import daily_chart_svg
    frame = pd.DataFrame({'in_kzt': [0.], 'out_kzt': [0.]}, index=pd.to_datetime(['2026-07-01']))
    svg = daily_chart_svg(frame)
    assert '<svg' in svg and '01.07' in svg
    assert '<script' not in svg and 'nan' not in svg.lower() and 'inf' not in svg.lower()
