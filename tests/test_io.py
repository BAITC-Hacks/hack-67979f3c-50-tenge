"""Входной контракт и сохранение структуры полного графа."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pipeline.features import compute_features
from pipeline.graph import build_graph
from pipeline.io import load_data, validate_inputs


def sample_data():
    nodes = pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 1, 0], "is_seed": [True, False, True]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "sum_kzt": [12000.0], "n_tx": [2], "depth": [1]})
    tx = pd.DataFrame({"src": [1, 1], "dst": [2, 2], "sum_kzt": [5000.0, 7000.0],
                       "date": ["2026-07-01", "2026-07-02"]})
    return nodes, edges, tx


def test_roundtrip_and_no_mutation(tmp_path):
    original = sample_data()
    for name, frame in zip(("nodes", "edges", "transactions"), original):
        frame.to_parquet(tmp_path / f"{name}.parquet")
    frames = load_data(tmp_path)
    snapshots = [frame.copy(deep=True) for frame in frames]
    validate_inputs(*frames)
    assert pd.api.types.is_datetime64_any_dtype(frames[2].date)
    for frame, snapshot in zip(frames, snapshots):
        pd.testing.assert_frame_equal(frame, snapshot)


@pytest.mark.parametrize("case,match", [
    ("missing_column", "отсутствуют колонки"),
    ("duplicate_gid", "повторяющиеся gid"),
    ("duplicate_pair", "повторяющиеся пары"),
    ("unknown_endpoint", "отсутствуют в nodes"),
    ("null", "пропуски"),
    ("infinite", "конечные положительные"),
    ("negative", "конечные положительные"),
    ("zero", "конечные положительные"),
    ("fractional_count", "целочисленный тип"),
    ("zero_count", "положительные целые"),
    ("float_gid", "целочисленный тип"),
    ("string_seed", "булев тип"),
    ("bad_depth", "диапазон"),
    ("bad_date", "некорректная дата"),
    ("different_pair", "не совпадают пары"),
    ("different_count", "не совпадают количества"),
    ("different_sum", "не совпадают суммы"),
])
def test_invalid_inputs(case, match):
    nodes, edges, tx = sample_data()
    if case == "missing_column":
        nodes = nodes.drop(columns="depth")
    elif case == "duplicate_gid":
        nodes = pd.concat([nodes, nodes.iloc[:1]])
    elif case == "duplicate_pair":
        edges = pd.concat([edges, edges])
    elif case == "unknown_endpoint":
        edges.loc[0, "dst"] = 99
    elif case == "null":
        tx.loc[0, "sum_kzt"] = np.nan
    elif case in ("infinite", "negative", "zero"):
        tx.loc[0, "sum_kzt"] = {"infinite": np.inf, "negative": -1, "zero": 0}[case]
    elif case == "fractional_count":
        edges["n_tx"] = 1.5
    elif case == "zero_count":
        edges["n_tx"] = 0
    elif case == "float_gid":
        nodes["gid"] = nodes.gid.astype(float)
    elif case == "string_seed":
        nodes["is_seed"] = "False"
    elif case == "bad_depth":
        nodes.loc[0, "depth"] = 5
    elif case == "bad_date":
        tx.loc[0, "date"] = "not-a-date"
    elif case == "different_pair":
        tx["dst"] = 3
    elif case == "different_count":
        edges["n_tx"] = 3
    elif case == "different_sum":
        edges.loc[0, "sum_kzt"] += 0.01
    with pytest.raises(ValueError, match=match):
        validate_inputs(nodes, edges, tx)


def test_float_aggregation_tolerance():
    nodes, edges, tx = sample_data()
    edges.loc[0, "sum_kzt"] += 1e-10
    validate_inputs(nodes, edges, tx)


def test_identical_transfers_are_not_deduplicated():
    nodes, edges, tx = sample_data()
    tx = pd.concat([tx.iloc[:1], tx.iloc[:1]], ignore_index=True)
    edges["sum_kzt"] = 10000.0
    validate_inputs(nodes, edges, tx)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="nodes.parquet"):
        load_data(tmp_path)


def test_complete_graph_direction_weights_and_isolate():
    nodes, edges, _ = sample_data()
    graph = build_graph(nodes, edges)
    assert set(graph) == {1, 2, 3}
    assert list(graph.edges) == [(1, 2)]
    assert graph[1][2] == {"sum_kzt": 12000.0, "n_tx": 2, "depth": 1}
    assert graph.nodes[3] == {"depth": 0, "is_seed": True}
    frame = compute_features(graph, nodes).set_index("gid")
    assert frame.loc[1, "out_kzt"] == frame.loc[2, "in_kzt"] == 12000
    assert frame.loc[1, "out_tx"] == frame.loc[2, "in_tx"] == 2
    assert frame.loc[1, "out_avg_kzt"] == 6000
    assert np.isnan(frame.loc[1, "pass_through"])
    assert np.isnan(frame.loc[3, "in_avg_kzt"])
    assert frame.loc[2, "pass_through"] == 0
    assert frame.loc[1, "wcc_id"] == frame.loc[2, "wcc_id"] != frame.loc[3, "wcc_id"]
    assert frame.pagerank.sum() == pytest.approx(1)


def test_features_betweenness_uses_hops_and_pagerank_uses_money():
    nodes = pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 1, 2], "is_seed": [True, False, False]})
    edges = pd.DataFrame({"src": [1, 2, 1], "dst": [2, 3, 3], "sum_kzt": [1., 1., 100.],
                          "n_tx": [1, 1, 1], "depth": [1, 2, 1]})
    graph = build_graph(nodes, edges)
    result = compute_features(graph, nodes).set_index("gid")
    assert result.loc[2, "betweenness"] == 0  # Direct edge is one hop, despite its large sum.
    assert nx.betweenness_centrality(graph, weight="sum_kzt")[2] > 0
    unweighted_pr = nx.pagerank(graph, weight=None)
    assert result.loc[2, "pagerank"] != pytest.approx(unweighted_pr[2])


def test_stable_graph_and_features_when_input_is_shuffled():
    nodes, edges, _ = sample_data()
    original = compute_features(build_graph(nodes, edges), nodes).sort_values("gid").reset_index(drop=True)
    shuffled = nodes.iloc[::-1]
    result = compute_features(build_graph(shuffled, edges.iloc[::-1]), shuffled).sort_values("gid").reset_index(drop=True)
    pd.testing.assert_frame_equal(result, original)


def test_graph_rejects_silent_node_insertion_or_edge_overwrite():
    nodes, edges, _ = sample_data()
    with pytest.raises(ValueError, match="уникальны"):
        build_graph(nodes, pd.concat([edges, edges]))
    with pytest.raises(ValueError, match="концы рёбер"):
        build_graph(nodes.iloc[:1], edges)


def test_features_require_full_directed_graph():
    nodes, edges, _ = sample_data()
    graph = build_graph(nodes, edges)
    with pytest.raises(ValueError, match="DiGraph"):
        compute_features(graph.to_undirected(), nodes)
    graph.remove_node(3)
    with pytest.raises(ValueError, match="полный граф"):
        compute_features(graph, nodes)


def test_real_input_and_component_counts():
    root = Path(__file__).resolve().parents[1]
    nodes, edges, tx = load_data(root / "data")
    validate_inputs(nodes, edges, tx)
    graph = build_graph(nodes, edges)
    assert (len(graph), graph.number_of_edges(), len(tx)) == (2248, 3119, 4840)
    assert nx.number_weakly_connected_components(graph) == 35
    isolates = list(nx.isolates(graph))
    assert len(isolates) == 19
    assert nodes.set_index("gid").loc[isolates, "is_seed"].all()
