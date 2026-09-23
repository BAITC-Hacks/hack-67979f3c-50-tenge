import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pipeline.seeds import compute_seed_reach
from pipeline.clusters import assign_clusters, summarize_clusters, _undirected_projection
from pipeline.priority import compute_priority, make_top_nodes


def test_seed_paths_cutoff_cycles_isolates_and_input_order():
    graph = nx.DiGraph([(1, 2), (2, 1), (2, 3), (3, 4), (4, 5), (5, 6)])
    graph.add_nodes_from([7, 8])
    nodes = pd.DataFrame({"gid": [8, 7, 6, 5, 4, 3, 2, 1],
                          "is_seed": [False, True, False, False, False, False, True, True]})
    original = nodes.copy(deep=True)
    result = compute_seed_reach(graph, nodes).set_index("gid")
    assert result.loc[1, "seed_reach_count"] == 1
    assert result.loc[2, "seed_reach_count"] == 1
    assert result.loc[3, "seed_reach_count"] == 2
    assert result.loc[6, "seed_reach_count"] == 1
    assert result.loc[6, "seed_distance"] == 4
    assert result.loc[7, "seed_distance"] == 0
    assert result.loc[7, "seed_reach_count"] == 0
    assert pd.isna(result.loc[8, "seed_distance"])
    assert_frame_equal(nodes, original)


def test_seed_reverse_direction_and_missing_isolate():
    graph = nx.DiGraph([(2, 1)])
    nodes = pd.DataFrame({"gid": [1, 2], "is_seed": [True, False]})
    result = compute_seed_reach(graph, nodes).set_index("gid")
    assert result.loc[2, "seed_reach_count"] == 0
    assert pd.isna(result.loc[2, "seed_distance"])
    graph.remove_node(2)
    with pytest.raises(ValueError):
        compute_seed_reach(graph, nodes)


def test_clusters_reciprocal_weights_isolates_determinism():
    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2, 3, 4, 5])
    graph.add_weighted_edges_from([(1, 2, 10), (2, 1, 30), (3, 4, 20)], weight="sum_kzt")
    nodes = pd.DataFrame({"gid": [1, 2, 3, 4, 5]})
    assert _undirected_projection(graph)[1][2]["sum_kzt"] == 40
    clusters = assign_clusters(graph, nodes)
    mapping = clusters.set_index("gid").cluster_id
    assert mapping[1] == mapping[2] == 0
    assert mapping[3] == mapping[4] == 1
    assert mapping[5] == 2
    reversed_graph = nx.DiGraph()
    reversed_graph.add_nodes_from(reversed(list(graph)))
    reversed_graph.add_edges_from(reversed(list(graph.edges(data=True))))
    assert_frame_equal(clusters, assign_clusters(reversed_graph, nodes.iloc[::-1]))
    scored = clusters.assign(is_seed=[True, False, False, False, True],
                             role="peripheral", priority_score=[.2, .7, .1, .9, 0])
    summary = summarize_clusters(graph, scored).set_index("cluster_id")
    assert summary.loc[0, "sum_kzt_internal"] == 40
    assert json.loads(summary.loc[0, "top_gids"]) == [2, 1]
    assert summary.loc[2, "n_seed"] == 1
    assert "Изолят" in summary.loc[2, "hypothesis"]


def priority_fixture():
    return pd.DataFrame({
        "gid": [10, 20, 30, 40], "is_seed": [True, False, False, False],
        "in_kzt": [100000, 10, 20, 0], "out_kzt": [10, 0, 0, 0],
        "in_deg": [100, 1, 1, 0], "out_deg": [1, 0, 0, 0],
        "betweenness": [0, 0, 1, 0], "seed_reach_count": [0, 1, 3, 0],
        "role": ["peripheral"]*4, "role_score": [.1]*4,
        "depth": [0, 4, 2, 0], "truncated_by_depth": [False, True, False, False],
    }, index=[5, 5, 2, 9])


def test_priority_exact_percentiles_seed_correction_and_no_mutation():
    source = priority_fixture()
    original = source.copy(deep=True)
    scored = compute_priority(source).set_index("gid")
    assert scored.loc[10, "priority_volume"] == pytest.approx(.35*.5)
    assert scored.loc[20, "priority_volume"] == pytest.approx(.35*.5)
    assert scored.loc[30, "priority_volume"] == pytest.approx(.35)
    assert scored.loc[10, "priority_degree"] == pytest.approx(.15*2/3)
    assert scored.loc[30, "priority_score"] == pytest.approx(.95)
    assert scored.loc[40, "priority_score"] == 0
    assert_frame_equal(source, original)
    top = make_top_nodes(scored.reset_index())
    assert list(top.gid) == [30, 20, 10, 40]
    assert "дальнейшие переводы неизвестны" in top.loc[top.gid == 20, "why"].iloc[0]


def test_priority_zero_features_and_nonfinite_rejection():
    source = priority_fixture()
    for col in ["in_kzt", "out_kzt", "in_deg", "out_deg", "betweenness", "seed_reach_count"]:
        source[col] = 0
    assert (compute_priority(source).priority_score == 0).all()
    source["in_kzt"] = source["in_kzt"].astype(float)
    source.iloc[0, source.columns.get_loc("in_kzt")] = float("inf")
    with pytest.raises(ValueError):
        compute_priority(source)


def test_top_seed_policy_and_tie_order():
    source = pd.concat([priority_fixture().iloc[[0]]]*40, ignore_index=True)
    source["gid"] = list(range(40, 0, -1))
    source["is_seed"] = [True]*11 + [False]*29
    source["in_kzt"] = source["out_kzt"] = 10
    source["in_deg"] = 0
    source["out_deg"] = 1
    scored = compute_priority(source)
    # Give 11 seeds a higher score with consistent contributions.
    scored.loc[scored.is_seed, "priority_bridge"] = .3
    scored["priority_score"] = scored[["priority_volume", "priority_bridge", "priority_seed", "priority_degree"]].sum(axis=1)
    top = make_top_nodes(scored)
    assert top.attrs["selection_policy"] == "non_seed"
    assert top.attrs["initial_top_seed_count"] == 11
    assert len(top) == 29
    assert list(top.gid) == list(range(1, 30))
    assert set(top.gid).isdisjoint(set(source.loc[source.is_seed, "gid"]))


def test_fractional_identifiers_cannot_silently_merge_clients():
    nodes = pd.DataFrame({"gid": [1.2, 1.8], "is_seed": [True, False]})
    graph = nx.DiGraph()
    graph.add_edge(1.2, 1.8, sum_kzt=5000)
    for function in [assign_clusters, compute_seed_reach]:
        with pytest.raises(ValueError, match="int64"):
            function(graph, nodes)
    features = priority_fixture()
    features["gid"] = features.gid.astype(float)
    with pytest.raises(ValueError, match="int64"):
        compute_priority(features)


def test_adjacent_large_ids_are_preserved_exactly():
    left, right = 2**63 - 2, 2**63 - 1
    nodes = pd.DataFrame({"gid": [left, right], "is_seed": [True, False]})
    graph = nx.DiGraph()
    graph.add_edge(left, right, sum_kzt=5000)
    assert set(assign_clusters(graph, nodes).gid) == {left, right}
    reach = compute_seed_reach(graph, nodes).set_index("gid")
    assert reach.loc[right, "seed_reach_count"] == 1
    features = priority_fixture().iloc[:2].copy()
    features["gid"] = [left, right]
    assert set(make_top_nodes(compute_priority(features)).gid).issubset({left, right})


def test_real_data_b_modules_and_export_contract(tmp_path):
    """Real flows; synthetic role labels ONLY to test the B export contract.

    This test does not classify clients or create deliverable AML results.
    Replace the fixture with A's assign_roles in the final integration run.
    """
    from starter.starter import basic_features
    from pipeline.outputs import write_outputs

    data = Path(__file__).resolve().parents[1] / "data"
    if not (data / "nodes.parquet").exists():
        pytest.skip("Данные хакатона отсутствуют")
    nodes = pd.read_parquet(data / "nodes.parquet")
    edges = pd.read_parquet(data / "edges.parquet")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.gid)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.src, row.dst, sum_kzt=float(row.sum_kzt), n_tx=int(row.n_tx))
    features = basic_features(graph, nodes)
    features["betweenness"] = features.gid.map(nx.betweenness_centrality(graph, weight=None))
    features = features.merge(compute_seed_reach(graph, nodes), on="gid", validate="one_to_one")
    clusters = assign_clusters(graph, nodes)
    features = features.merge(clusters, on="gid", validate="one_to_one")
    features["role"] = "peripheral"
    features["role_score"] = 0.1
    features["evidence"] = "Тест контракта B: 0 выводов о роли; ожидается модуль A"
    scored = compute_priority(features)
    top = make_top_nodes(scored)
    summary = summarize_clusters(graph, scored)
    assert len(scored) == 2248
    assert set(scored.gid) == set(nodes.gid)
    assert nx.number_of_isolates(graph) == 19
    assert nx.number_weakly_connected_components(graph) == 35
    assert len(top) == 30
    # Fixed algorithm settings and stable ordering must survive input shuffles.
    shuffled_graph = nx.DiGraph()
    shuffled_graph.add_nodes_from(reversed(list(graph)))
    shuffled_graph.add_edges_from(reversed(list(graph.edges(data=True))))
    assert_frame_equal(clusters, assign_clusters(shuffled_graph, nodes.iloc[::-1]))
    for row in summary.itertuples(index=False):
        members = set(scored.loc[scored.cluster_id == row.cluster_id, "gid"])
        assert nx.is_weakly_connected(graph.subgraph(members))
        mask = edges.src.isin(members) & edges.dst.isin(members)
        assert row.sum_kzt_internal == pytest.approx(edges.loc[mask, "sum_kzt"].sum())
    seed_ids = nodes.loc[nodes.is_seed, "gid"].tolist()
    for gid in nodes.gid:
        # Reverse BFS is independent of implementation's forward BFS per seed.
        ancestors = nx.single_source_shortest_path_length(graph.reverse(copy=False), gid, cutoff=4)
        expected = sum(s != gid and s in ancestors for s in seed_ids)
        assert scored.loc[scored.gid == gid, "seed_reach_count"].iloc[0] == expected
    write_outputs(scored, summary, top, tmp_path / "first")
    again = compute_priority(features.sample(frac=1, random_state=7))
    summary_again = summarize_clusters(graph, again)
    top_again = make_top_nodes(again)
    write_outputs(again, summary_again, top_again, tmp_path / "second")
    for name in ["nodes_roles.csv", "clusters.csv", "top_nodes.csv"]:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    # Exact large identifiers must survive CSV; never pass them through float.
    restored = pd.read_csv(tmp_path / "first" / "nodes_roles.csv")
    assert restored.gid.dtype == "int64"
    assert set(restored.gid) == set(nodes.gid)

    # Reproducible diagnostics: measure sensitivity, do not assert an arbitrary
    # quality threshold or tune the algorithm to a fixed expected cluster count.
    sensitivity = {"resolutions": [], "weights": [], "selection": dict(top.attrs)}
    for resolution in [.8, 1., 1.2]:
        mapping = assign_clusters(graph, nodes, resolution=resolution)
        sizes = mapping.groupby("cluster_id").size()
        sensitivity["resolutions"].append({"resolution": resolution,
                                           "clusters": len(sizes), "largest": int(sizes.max())})
    weights = {"priority_volume": .35, "priority_bridge": .30,
               "priority_seed": .20, "priority_degree": .15}
    for column, weight in weights.items():
        for factor in [.8, 1.2]:
            variant = scored.copy()
            normalizer = 1 + weight * (factor - 1)
            for contribution in weights:
                variant[contribution] *= (factor if contribution == column else 1) / normalizer
            variant["priority_score"] = variant[list(weights)].sum(axis=1)
            alternative = make_top_nodes(variant)
            sensitivity["weights"].append({"column": column, "factor": factor,
                "top30_overlap": len(set(top.gid) & set(alternative.gid)),
                "selection_policy": alternative.attrs["selection_policy"]})
    (tmp_path / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
