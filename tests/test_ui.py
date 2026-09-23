"""Analyst workflow checks; calculate genuine inputs into a temporary folder."""

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pyvis")
from streamlit.testing.v1 import AppTest
import viz.app as app


@pytest.fixture(scope="module")
def ui_data(tmp_path_factory):
    from run import main
    data = Path(__file__).resolve().parents[1] / "data"
    if not (data / "nodes.parquet").exists():
        pytest.skip("Данные хакатона отсутствуют")
    output = tmp_path_factory.mktemp("analyst-ui")
    main(data, output)
    return output, pd.read_csv(output / "nodes_roles.csv"), pd.read_parquet(data / "edges.parquet")


def test_full_filters_and_connected_graph(ui_data):
    _, nodes, edges = ui_data
    assert len(app.filtered_nodes(nodes, [], [], "Все")) == len(nodes)
    assert len(app.filtered_nodes(nodes, [], [], "Да")) == int(nodes.is_seed.sum())
    gid = int(nodes.sort_values("priority_score", ascending=False).gid.iloc[0])
    visible, hidden = app.neighborhood(edges, gid, 2)
    graph = nx.from_pandas_edgelist(edges, "src", "dst")
    graph.add_nodes_from(nodes.gid)
    assert gid in visible and len(visible) <= 200
    assert nx.is_connected(graph.subgraph(visible))
    assert len(visible) + hidden == len(nx.single_source_shortest_path_length(graph, gid, cutoff=2))


def test_analyst_search_roles_boundary_and_unknown(ui_data, monkeypatch):
    output, nodes, _ = ui_data
    monkeypatch.setattr(app, "OUT", output)
    app.load_tables.clear()
    screen = AppTest.from_string("import viz.app as app\napp.main()", default_timeout=30).run()
    assert not screen.exception
    highest = int(nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).gid.iloc[0])
    assert screen.selectbox(key="client_select").value == str(highest)
    screen.radio(key="graph_color").set_value("По группам").run()
    screen.radio(key="graph_hops").set_value(2).run()
    assert not screen.exception
    screen.selectbox(key="seed_filter").set_value("Да").run()
    assert len(screen.selectbox(key="client_select").options) == int(nodes.is_seed.sum())
    ids = nodes.groupby("role").gid.first().tolist()
    ids += [int(nodes.loc[nodes.truncated_by_depth, "gid"].iloc[0]),
            int(nodes.loc[(nodes.in_deg == 0) & (nodes.out_deg == 0), "gid"].iloc[0])]
    for gid in ids:
        screen.text_input(key="client_search").set_value(str(gid)).run()
        assert not screen.exception, f"Не открывается клиент {gid}: {screen.exception}"
    screen.text_input(key="client_search").set_value("999").run()
    assert not screen.exception
    assert screen.warning or screen.error


def test_display_keeps_ids_and_translates_roles(ui_data):
    _, nodes, edges = ui_data
    table = app.client_table(nodes)
    assert table["Клиент"].tolist() == nodes.gid.astype(str).tolist()
    assert table["Предполагаемая роль"].notna().all()
    assert not set(table.columns) & {"gid", "role", "priority_score", "is_seed"}
    gid = int(nodes.gid.iloc[0])
    html, _ = app.graph_html(edges, nodes, gid, 1, "По группам")
    assert str(gid) in html
    assert "группа" in html or "\\u0433\\u0440\\u0443\\u043f\\u043f\\u0430" in html
