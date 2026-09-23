import json

import numpy as np
import pandas as pd
import pytest

from pipeline.outputs import write_outputs


@pytest.fixture
def tables():
    nodes = pd.DataFrame({
        "gid": [20, 10], "role": ["peripheral", "consolidator"],
        "role_score": [0.1, 0.6], "cluster_id": [1, 0],
        "priority_score": [0.0, 0.5], "evidence": ["0 связей", "Вход от 5 клиентов"],
        "is_seed": [True, False], "pass_through": [np.nan, 0.2],
    })
    clusters = pd.DataFrame({
        "cluster_id": [1, 0], "n_nodes": [1, 1], "n_seed": [1, 0],
        "sum_kzt_internal": [0., 0.], "top_gids": [json.dumps([20]), json.dumps([10])],
        "hypothesis": ["Изолят: 1 seed", "Группа: 1 клиент"],
    })
    top = pd.DataFrame({"rank": [1, 2], "gid": [10, 20],
                        "role": ["consolidator", "peripheral"],
                        "priority_score": [0.5, 0.], "why": ["Вход от 5 клиентов", "0 связей"]})
    return nodes, clusters, top


def test_round_trip_determinism_and_no_mutation(tables, tmp_path):
    before = [table.copy(deep=True) for table in tables]
    write_outputs(*tables, tmp_path / "first")
    write_outputs(tables[0].iloc[::-1], tables[1].iloc[::-1], tables[2], tmp_path / "second")
    for path in (tmp_path / "first").glob("*.csv"):
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()
    restored = pd.read_csv(tmp_path / "first" / "nodes_roles.csv")
    assert restored.gid.tolist() == [10, 20]
    assert pd.isna(restored.loc[restored.gid == 20, "pass_through"]).all()
    for original, snapshot in zip(tables, before):
        pd.testing.assert_frame_equal(original, snapshot)


@pytest.mark.parametrize("index,column,row,value", [
    (0, "role_score", 0, np.inf), (0, "role_score", 0, -0.1),
    (0, "evidence", 0, ""), (0, "evidence", 0, "Нет чисел"),
    (0, "evidence", 0, "1" * 201), (0, "role", 0, "guilty"),
    (0, "gid", 0, 10), (1, "n_nodes", 0, 2),
    (1, "n_seed", 0, 0), (1, "top_gids", 0, "[10]"),
    (1, "sum_kzt_internal", 0, np.inf), (1, "top_gids", 0, "invalid"),
    (2, "rank", 0, 2), (2, "gid", 0, 99),
    (2, "priority_score", 0, 0.6), (2, "role", 0, "transit"),
])
def test_invalid_tables_fail_before_writing(tables, tmp_path, index, column, row, value):
    tables[index].loc[row, column] = value
    with pytest.raises(ValueError):
        write_outputs(*tables, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_column(tables, tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        write_outputs(tables[0].drop(columns="evidence"), *tables[1:], tmp_path)


def test_too_short_top(tables, tmp_path):
    with pytest.raises(ValueError, match="at least"):
        write_outputs(tables[0], tables[1], tables[2].iloc[:1], tmp_path)


def test_identifiers_require_integer_dtype(tables, tmp_path):
    tables[0]["gid"] = tables[0].gid.astype(float)
    with pytest.raises(ValueError, match="integer dtype"):
        write_outputs(*tables, tmp_path)


def test_forbidden_boundary_terminal(tables, tmp_path):
    nodes, clusters, top = tables
    nodes["depth"], nodes["out_deg"] = [4, 1], [0, 1]
    nodes.loc[0, "role"] = "terminal"
    top.loc[1, "role"] = "terminal"
    with pytest.raises(ValueError, match="boundary"):
        write_outputs(nodes, clusters, top, tmp_path)


def test_inconsistent_contributions(tables, tmp_path):
    for col in ["priority_volume", "priority_bridge", "priority_seed", "priority_degree"]:
        tables[0][col] = 0.
    with pytest.raises(ValueError, match="sum mismatch"):
        write_outputs(*tables, tmp_path)


def test_ranking_ties_are_ordered_by_gid(tables, tmp_path):
    tables[0]["priority_score"] = 0.
    tables[2]["priority_score"] = 0.
    tables[2].loc[:, "gid"] = [20, 10]
    with pytest.raises(ValueError, match="order"):
        write_outputs(*tables, tmp_path)
