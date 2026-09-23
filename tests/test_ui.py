"""Analyst workflow checks; calculate genuine inputs into a temporary folder."""

from pathlib import Path
import json

import networkx as nx
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pyvis")
from streamlit.testing.v1 import AppTest
import viz.app as app


@pytest.fixture(autouse=True)
def no_live_api(monkeypatch):
    import viz.ai_assistant as assistant
    import viz.graph_agent as graph_agent

    def unexpected_request(*args, **kwargs):
        pytest.fail("Проверка интерфейса не должна обращаться к платному API")

    monkeypatch.setattr(assistant, "read_config", lambda: {})
    monkeypatch.setattr(graph_agent, "read_config", lambda: {})
    monkeypatch.setattr(assistant, "request_explanation", unexpected_request)
    monkeypatch.setattr(graph_agent, "urlopen", unexpected_request)


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


def test_group_selection_and_new_user_help(ui_data, monkeypatch):
    output, nodes, _ = ui_data
    monkeypatch.setattr(app, "OUT", output)
    app.load_tables.clear()
    screen = AppTest.from_string("import viz.app as app\napp.main()", default_timeout=30).run()
    assert not screen.exception
    selected_client = int(screen.selectbox(key="client_select").value)
    cluster_id = int(nodes.loc[nodes.gid.eq(selected_client), "cluster_id"].iloc[0])
    all_members = app.group_members(nodes, cluster_id)
    assert len(all_members) >= 2

    screen.toggle(key="show_help").set_value(True).run()
    assert not screen.exception
    assert any("С чего начать" in item.value for item in screen.markdown)

    screen.radio(key="group_mode").set_value("Выбрать клиентов").run()
    ids = all_members.gid.astype(str).tolist()[:2]
    screen.multiselect(key=f"group_clients_{cluster_id}").set_value(ids).run()
    assert not screen.exception
    assert set(app.group_members(nodes, cluster_id, ids).gid.astype(str)) == set(ids)
    assert len(app.group_members(nodes, cluster_id, ids[:1])) == 1
    assert len(app.group_members(nodes, cluster_id)) == len(all_members)


def _screen(ui_data, monkeypatch):
    monkeypatch.setattr(app, "OUT", ui_data[0])
    app.load_tables.clear()
    screen = AppTest.from_string("import viz.app as app\napp.main()", default_timeout=45).run()
    assert not screen.exception
    return screen


def test_filters_reset_and_exact_search_override(ui_data, monkeypatch):
    screen = _screen(ui_data, monkeypatch)
    _, nodes, _ = ui_data
    selected = nodes.loc[~nodes.is_seed].sort_values(["priority_score", "gid"], ascending=[False, True]).iloc[0]
    screen.multiselect(key="role_filter").set_value([selected.role]).run()
    screen.multiselect(key="cluster_filter").set_value([int(selected.cluster_id)]).run()
    screen.selectbox(key="seed_filter").set_value("Нет").run()
    eligible = nodes.loc[nodes.role.eq(selected.role) & nodes.cluster_id.eq(selected.cluster_id) & ~nodes.is_seed]
    assert len(screen.selectbox(key="client_select").options) == len(eligible)
    outside = str(int(nodes.loc[nodes.is_seed, "gid"].iloc[0]))
    screen.text_input(key="client_search").set_value(outside).run()
    assert not screen.exception
    assert screen.selectbox(key="client_select").disabled
    assert any(outside in text.value and "money-id" in text.value for text in screen.markdown)
    screen.button(key="clear_search").click().run()
    assert screen.text_input(key="client_search").value == ""
    assert not screen.selectbox(key="client_select").disabled
    screen.button(key="reset_filters").click().run()
    assert not screen.exception
    assert screen.multiselect(key="role_filter").value == []
    assert screen.multiselect(key="cluster_filter").value == []
    assert screen.selectbox(key="seed_filter").value == "Все"
    assert len(screen.selectbox(key="client_select").options) == len(nodes)


def test_offline_common_recipients_and_client_navigation(ui_data, monkeypatch):
    screen = _screen(ui_data, monkeypatch)
    _, nodes, edges = ui_data
    target_counts = edges.groupby("dst").src.nunique()
    target = target_counts.loc[target_counts >= 2].sort_index().index[0]
    sources = sorted(edges.loc[edges.dst.eq(target), "src"].unique())[:2]
    expected = set(edges.loc[edges.src.eq(sources[0]), "dst"]) & set(edges.loc[edges.src.eq(sources[1]), "dst"])
    expected -= set(sources)
    assert expected
    screen.text_input(key="graph_local_gids").set_value(", ".join(str(gid) for gid in sources)).run()
    screen.button(key="graph_local_run").click().run()
    assert not screen.exception
    result = screen.session_state["graph_local_result"]["trace"][0]["result"]
    assert result["total_matches"] == len(expected)
    assert set(result["sources"]) == {str(gid) for gid in sources}
    recipient = result["recipients"][0]
    assert int(recipient["gid"]) in expected
    for route in recipient["paths_from_each_source"]:
        assert route["path"][-1] == recipient["gid"]
        assert route["hops"] == 1
        assert len(route["edges"]) == 1
        edge = route["edges"][0]
        actual = edges.loc[edges.src.eq(int(edge["src"])) & edges.dst.eq(int(edge["dst"]))].iloc[0]
        assert edge["sum_kzt"] == actual.sum_kzt
        assert edge["n_tx"] == actual.n_tx
    screen.button(key=f"graph_local_node_{recipient['gid']}").click().run()
    assert not screen.exception
    assert screen.text_input(key="client_search").value == recipient["gid"]
    assert any("AI не подключён" in item.value for item in screen.info)
    assert not any(button.key in {"graph_agent_run", "ai_explain"} for button in screen.button)


def test_bonus_panels_real_reports_and_cluster_edge_cases(ui_data, monkeypatch):
    output, nodes, _ = ui_data
    screen = _screen(ui_data, monkeypatch)
    network = json.loads((output / "network_patterns.json").read_text())
    temporal = json.loads((output / "temporal_patterns.json").read_text())
    assert temporal["n_nodes"] == len(nodes)
    screen.radio(key="bonus_route_scope").set_value("По всей сети").run()
    assert not screen.exception
    for name in ("recurring_routes", "short_cycles"):
        if network[name]["items"]:
            widget = screen.selectbox(key=f"bonus_{name}_select")
            assert len(widget.options) == len(network[name]["items"])
            widget.set_value(len(network[name]["items"]) - 1).run()
            assert not screen.exception
    assert len(screen.selectbox(key="bonus_removal").options) == len(network["resilience"]["scenarios"])
    screen.selectbox(key="bonus_removal").set_value(len(network["resilience"]["scenarios"]) - 1).run()
    assert not screen.exception
    assert any("20 случайными" in item.value for item in screen.caption)
    clusters = pd.read_csv(output / "clusters.csv")
    uncertain = clusters.loc[clusters.stability_status.isin(["unstable", "variable"])]
    assert len(uncertain), "Настоящая выборка должна содержать группы для проверки устойчивости"
    for cluster_id in uncertain.cluster_id.head(2):
        gid = str(int(nodes.loc[nodes.cluster_id.eq(cluster_id), "gid"].iloc[0]))
        screen.text_input(key="client_search").set_value(gid).run()
        assert not screen.exception
        assert any("Сходство состава" in metric.label for metric in screen.metric)
    isolated = nodes.loc[nodes.in_deg.eq(0) & nodes.out_deg.eq(0)].iloc[0]
    screen.text_input(key="client_search").set_value(str(int(isolated.gid))).run()
    assert not screen.exception
    assert any("операций этого клиента нет" in item.value for item in screen.info)
    assert any("Изолированных клиентов" in item.value for item in screen.caption)


def test_collectors_shortcut_and_secondary_signals(ui_data, monkeypatch):
    output, nodes, _ = ui_data
    monkeypatch.setattr(app, 'OUT', output)
    app.load_tables.clear()
    screen = AppTest.from_string('import viz.app as app\napp.main()', default_timeout=30).run()
    screen.selectbox(key='seed_filter').set_value('Да').run()
    screen.text_input(key='client_search').set_value(str(int(nodes.gid.iloc[0]))).run()
    screen.button(key='open_collectors').click().run()
    assert not screen.exception
    assert screen.text_input(key='client_search').value == ''
    assert screen.selectbox(key='seed_filter').value == 'Все'
    assert screen.multiselect(key='role_filter').value == ['consolidator']
    expected = nodes[nodes.role.eq('consolidator')].sort_values(
        ['priority_score', 'gid'], ascending=[False, True])
    assert screen.selectbox(key='client_select').value == str(int(expected.gid.iloc[0]))
    assert len(screen.selectbox(key='client_select').options) == len(expected)
    metadata = json.loads((output / 'run_metadata.json').read_text())
    overlapping = nodes[nodes.role.eq('coordinator') & nodes.out_deg.ge(5)
                        & nodes.out_deg.ge(2 * nodes.in_deg.clip(lower=1))]
    assert len(overlapping) > 0
    row = overlapping.iloc[0]
    assert 'distributor' in app.secondary_signals(row, metadata)
    screen.text_input(key='client_search').set_value(str(int(row.gid))).run()
    assert not screen.exception
    assert any('Дополнительные признаки' in item.value for item in screen.markdown)
    for _, boundary in nodes[nodes.truncated_by_depth].iterrows():
        assert 'terminal' not in app.secondary_signals(boundary, metadata)
