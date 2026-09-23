"""Правила, границы применимости и формулы оценок потока A."""

from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pipeline.features import compute_features
from pipeline.graph import build_graph
from pipeline.io import load_data, validate_inputs
from pipeline.roles import ROLES, assign_roles, compute_role_thresholds


def features(*overrides):
    rows = []
    for gid, override in enumerate(overrides, 1):
        row = {"gid": gid, "depth": 1, "is_seed": False, "in_deg": 1, "out_deg": 1,
               "in_kzt": 100., "out_kzt": 100., "betweenness": 0.,
               "seed_reach_count": 1, "seed_distance": 1.}
        row.update(override)
        row["pass_through"] = row["out_kzt"] / row["in_kzt"] if row["in_kzt"] else np.nan
        row["truncated_by_depth"] = row["depth"] == 4 and row["out_deg"] == 0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("override,rule,role,score", [
    ({"in_deg": 0, "out_deg": 0, "in_kzt": 0., "out_kzt": 0.}, "isolated", "peripheral", .10),
    ({"depth": 4, "out_deg": 0, "out_kzt": 0.}, "boundary_unknown", "peripheral", .15),
    ({"depth": 4, "in_deg": 3, "out_deg": 0, "out_kzt": 0.}, "boundary_consolidator", "consolidator", .40),
    ({"seed_reach_count": 2, "betweenness": .1}, "coordinator", "coordinator", .45),
    ({"in_deg": 3}, "consolidator", "consolidator", .55),
    ({"out_deg": 5}, "distributor", "distributor", .55),
    ({"depth": 3, "out_deg": 0, "out_kzt": 0.}, "terminal", "terminal", .55),
    ({}, "transit", "transit", .65),
    ({"out_kzt": 200.}, "fallback", "peripheral", .20),
])
def test_each_rule_at_threshold(override, rule, role, score):
    result = assign_roles(features(override)).iloc[0]
    assert result.role_rule == rule
    assert result.role == role
    assert result.role_score == pytest.approx(score)
    assert 0 < len(result.evidence) <= 200
    assert any(character.isdigit() for character in result.evidence)


def test_boundary_collector_has_same_input_evidence_and_lower_score():
    frame = features({"in_deg": 9, "out_deg": 0, "out_kzt": 0., "depth": 3},
                     {"in_deg": 9, "out_deg": 0, "out_kzt": 0., "depth": 4})
    result = assign_roles(frame)
    assert result.role.tolist() == ["consolidator", "consolidator"]
    assert result.loc[0, "role_score"] - result.loc[1, "role_score"] == pytest.approx(.15)
    assert "обрыв обхода" in result.loc[1, "evidence"]
    assert "неизвестен" in result.loc[1, "evidence"]


def test_same_leaf_topology_depth_three_and_four():
    result = assign_roles(features({"out_deg": 0, "out_kzt": 0., "depth": 3},
                                    {"out_deg": 0, "out_kzt": 0., "depth": 4}))
    assert result.role.tolist() == ["terminal", "peripheral"]
    assert result.role_score.tolist() == [.55, .15]


@pytest.mark.parametrize("in_degree,in_kzt", [(0, 0.), (1, 1.), (100, 100000.)])
def test_seed_role_and_score_do_not_depend_on_observed_inflow(in_degree, in_kzt):
    result = assign_roles(features({"is_seed": True, "depth": 0, "in_deg": in_degree,
                                    "in_kzt": in_kzt, "out_deg": 10, "out_kzt": 1000.})).iloc[0]
    assert result.role == "distributor"
    assert result.role_score == pytest.approx(.70)
    assert "неполный вход не используется" in result.evidence


@pytest.mark.parametrize("out_degree", [0, 1, 4])
def test_seed_never_gets_inflow_based_role(out_degree):
    result = assign_roles(features({"is_seed": True, "in_deg": 20, "in_kzt": 100000.,
                                    "out_kzt": 100000., "out_deg": out_degree,
                                    "seed_reach_count": 5, "betweenness": .5})).iloc[0]
    assert result.role_rule == "fallback"


def test_empty_quantile_samples_disable_corresponding_rules():
    frame = features({"is_seed": True, "out_deg": 5},
                     {"in_kzt": 0., "out_kzt": 100., "in_deg": 0, "out_deg": 1})
    assert compute_role_thresholds(frame) == {"in_kzt_q75": None, "betweenness_q95": None}
    result = assign_roles(frame)
    assert result.role.tolist() == ["distributor", "peripheral"]
    assert np.isfinite(result.role_score).all()


def test_quantiles_exclude_seed_inflows_and_zero_betweenness():
    frame = features({"in_kzt": 100., "betweenness": 0.},
                     {"in_kzt": 200., "betweenness": 1.},
                     {"in_kzt": 300., "betweenness": 3.},
                     {"in_kzt": 400., "betweenness": 0.},
                     {"is_seed": True, "in_kzt": 1e9, "betweenness": 0.})
    assert compute_role_thresholds(frame) == {"in_kzt_q75": 325., "betweenness_q95": 2.9}


def test_score_changes_with_strength_inside_same_role():
    result = assign_roles(features({"out_deg": 5}, {"out_deg": 10}, {"out_deg": 15},
                                    {"out_deg": 100}))
    assert result.role.eq("distributor").all()
    assert result.role_score.tolist() == pytest.approx([.55, .70, .85, .85])


def test_collector_score_strength_and_saturation():
    frame = features(*([{}] * 8), {"in_deg": 3, "in_kzt": 100.},
                     {"in_deg": 9, "in_kzt": 300.})
    result = assign_roles(frame)
    assert result.attrs["role_thresholds"]["in_kzt_q75"] == 100.
    assert result.iloc[-2:].role_score.tolist() == pytest.approx([.55, .85])


def test_coordinator_score_strength_and_saturation():
    frame = features(*([{"betweenness": .1}] * 20),
                     {"betweenness": .1, "seed_reach_count": 2},
                     {"betweenness": .3, "seed_reach_count": 6})
    result = assign_roles(frame)
    assert result.attrs["role_thresholds"]["betweenness_q95"] == pytest.approx(.1)
    assert result.iloc[-2:].role_score.tolist() == pytest.approx([.45, .75])


@pytest.mark.parametrize("ratio,role,score", [(.699, "peripheral", .2), (.7, "transit", .45),
                                             (1., "transit", .65), (1.3, "transit", .45),
                                             (1.301, "peripheral", .2)])
def test_transit_ratio_boundaries(ratio, role, score):
    result = assign_roles(features({"out_kzt": ratio * 100})).iloc[0]
    assert result.role == role
    assert result.role_score == pytest.approx(score)


def test_precedence_for_overlapping_rules():
    frame = features({"in_deg": 9, "out_deg": 1, "betweenness": .1, "seed_reach_count": 2},
                     {"in_deg": 3, "out_deg": 0, "out_kzt": 0.},
                     {"out_deg": 5})
    result = assign_roles(frame)
    assert result.role.tolist() == ["coordinator", "consolidator", "distributor"]
    assert result.attrs["role_candidate_counts"]["transit"] == 2


def test_collector_and_distributor_degree_ratio_boundaries():
    frame = features({"in_deg": 3, "out_deg": 2, "out_kzt": 10.},
                     {"in_deg": 4, "out_deg": 2, "out_kzt": 10.},
                     {"in_deg": 3, "out_deg": 5, "out_kzt": 10.},
                     {"in_deg": 3, "out_deg": 6, "out_kzt": 10.})
    assert assign_roles(frame).role.tolist() == ["peripheral", "consolidator", "peripheral", "distributor"]


def test_no_mutation_and_no_index_based_assignment():
    frame = features({}, {"out_deg": 5})
    frame.index = [99, 99]
    original = frame.copy(deep=True)
    result = assign_roles(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert result.gid.tolist() == frame.gid.tolist()
    assert result.role.tolist() == ["transit", "distributor"]


def test_missing_seed_metrics_is_an_integration_error():
    with pytest.raises(ValueError, match="seed-метрики B"):
        assign_roles(features({}).drop(columns="seed_reach_count"))


def test_no_missing_scores_after_bad_seed_join():
    frame = features({})
    frame["seed_reach_count"] = np.nan
    with pytest.raises(ValueError, match="seed_reach_count"):
        assign_roles(frame)


def test_boundary_flag_must_match_topology():
    frame = features({"depth": 4, "out_deg": 0})
    frame["truncated_by_depth"] = False
    with pytest.raises(ValueError, match="truncated_by_depth"):
        assign_roles(frame)


def test_short_evidence_preserves_caveats_for_large_numbers():
    frame = features({"in_deg": 100000, "depth": 4, "out_deg": 0, "in_kzt": 1e100},
                     {"seed_reach_count": 100000, "betweenness": 1e-100},
                     {"in_kzt": 1e100, "out_kzt": 1e100})
    result = assign_roles(frame)
    assert result.evidence.str.len().le(200).all()
    assert result.role.isin(ROLES).all()


def test_chain_from_graph_to_roles():
    nodes = pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 1, 2], "is_seed": [True, False, False]})
    edges = pd.DataFrame({"src": [1, 2], "dst": [2, 3], "sum_kzt": [100., 100.],
                          "n_tx": [1, 1], "depth": [1, 2]})
    frame = compute_features(build_graph(nodes, edges), nodes)
    # Explicit fixture for B's contract, not a production replacement of seeds.py.
    seed_metrics = pd.DataFrame({"gid": [3, 1, 2], "seed_reach_count": [1, 0, 1], "seed_distance": [2., 0., 1.]})
    result = assign_roles(frame.merge(seed_metrics, on="gid", validate="one_to_one"))
    assert result.role.tolist() == ["peripheral", "transit", "terminal"]


def test_real_roles_contract_and_repeatability():
    """A-only integration: reverse BFS is an explicit test fixture for B.

    This does not validate the production seeds.py or the full three-CSV run.
    """
    start = time.perf_counter()
    nodes, edges, tx = load_data(Path(__file__).resolve().parents[1] / "data")
    validate_inputs(nodes, edges, tx)
    graph = build_graph(nodes, edges)
    frame = compute_features(graph, nodes)
    seeds = set(nodes.loc[nodes.is_seed, "gid"])
    reverse = graph.reverse(copy=False)
    records = []
    for gid in nodes.gid:
        distances = nx.single_source_shortest_path_length(reverse, gid, cutoff=4)
        sources = seeds.intersection(distances) - {gid}
        distance = 0 if gid in seeds else min((distances[s] for s in sources), default=np.nan)
        records.append((gid, len(sources), distance))
    seed_df = pd.DataFrame(records, columns=["gid", "seed_reach_count", "seed_distance"])
    result = assign_roles(frame.merge(seed_df, on="gid", validate="one_to_one"))
    assert len(result) == 2248 and result.gid.is_unique
    assert set(result.gid) == set(nodes.gid)
    required = ["role", "role_rule", "role_score", "evidence"]
    assert not result[required].isna().any().any()
    assert result.role.isin(ROLES).all()
    assert result.role_score.between(0, 1).all() and np.isfinite(result.role_score).all()
    assert result.evidence.str.len().between(1, 200).all()
    assert result.evidence.str.contains(r"\d").all()
    boundary = result[result.truncated_by_depth]
    assert len(boundary) == 444
    assert not boundary.role.eq("terminal").any()
    assert boundary.role_rule.isin(["boundary_consolidator", "boundary_unknown"]).all()
    assert result.role_rule.eq("isolated").sum() == 19
    shuffled_nodes = nodes.sample(frac=1, random_state=42)
    shuffled_edges = edges.sample(frac=1, random_state=42)
    repeated = assign_roles(compute_features(build_graph(shuffled_nodes, shuffled_edges), shuffled_nodes)
                            .merge(seed_df, on="gid", validate="one_to_one"))
    pd.testing.assert_frame_equal(result.sort_values("gid").reset_index(drop=True),
                                  repeated.sort_values("gid").reset_index(drop=True), check_exact=True)
    print({"roles": result.role.value_counts().to_dict(),
           "rules": result.role_rule.value_counts().to_dict(),
           "thresholds": result.attrs["role_thresholds"],
           "two_A_runs_seconds": round(time.perf_counter() - start, 3)})
