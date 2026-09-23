"""Local graph and mocked Responses checks; never call the live API."""
from contextlib import nullcontext
import io
import json

import pandas as pd
import pytest

from viz import graph_agent as agent
from viz.ai_transport import AssistantResponseError


@pytest.fixture
def tables():
    roles = pd.DataFrame({"gid": range(1, 9), "cluster_id": [0]*8,
        "role": ["peripheral"]*8, "priority_score": [0, 0, 0, .2, .9, .4, .3, 0]})
    edges = pd.DataFrame([(1, 4), (2, 4), (4, 5), (3, 5), (5, 4), (5, 6), (6, 7)], columns=["src", "dst"])
    edges["sum_kzt"], edges["n_tx"] = 5000., 1
    return roles, edges


def response(output):
    return {"status": "completed", "output": output, "usage": {"input_tokens": 10, "output_tokens": 4}}


def call(name="lookup_node", arguments=None, call_id="c1"):
    return {"type": "function_call", "name": name, "call_id": call_id,
            "arguments": json.dumps(arguments if arguments is not None else {"gid": "1"})}


def text_message(text="Клиент 1: роль не определена."):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def fake_api(monkeypatch, payloads):
    requests = []
    replies = iter(payloads)
    def open_fake(request, timeout):
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert 0 < timeout <= 20
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(next(replies)).encode())
    monkeypatch.setattr(agent, "urlopen", open_fake)
    return requests


def test_all_sources_required_and_cycle_cutoff(tables):
    graph = agent.GraphQueries(*tables)
    assert graph.common_recipients(["1", "2", "3"], 1)["total_matches"] == 0
    result = graph.common_recipients(["1", "2", "3"], 2)
    assert [row["gid"] for row in result["recipients"]] == ["5", "4"]
    for row in result["recipients"]:
        paths = row["paths_from_each_source"]
        assert {path["path"][0] for path in paths} == {"1", "2", "3"}
        assert all(path["hops"] <= 2 for path in paths)
        assert all(path["path"][-1] == row["gid"] for path in paths)
    assert not graph.find_path("1", "7", 3)["found"]
    assert graph.find_path("1", "7", 4)["route"]["path"] == ["1", "4", "5", "6", "7"]
    assert not graph.find_path("7", "1", 4)["found"]
    assert graph.common_recipients(["4", "5"], 4)["sources"] == ["4", "5"]
    assert all(row["gid"] not in {"4", "5"} for row in graph.common_recipients(["4", "5"], 4)["recipients"])


@pytest.mark.parametrize("name,args", [
    ("lookup_node", {"gid": "999"}), ("lookup_node", {"gid": 1}),
    ("lookup_node", {"gid": "1", "extra": True}), ("eval", {"code": "1+1"}),
    ("common_recipients", {"gids": ["1", "1"], "max_hops": 2}),
    ("common_recipients", {"gids": ["1"], "max_hops": 2}),
    ("find_path", {"src": "1", "dst": "7", "max_hops": True}),
    ("find_path", {"src": "1", "dst": "7", "max_hops": 5}),
    ("rank_cluster", {"cluster_id": 999, "limit": 20}),
    ("rank_cluster", {"cluster_id": 0, "limit": 21}),
])
def test_invalid_tool_arguments(tables, name, args):
    with pytest.raises(AssistantResponseError):
        agent.GraphQueries(*tables).dispatch(name, args)


def test_long_ids_remain_exact(tables):
    roles, edges = tables
    shift = 2**63 - 10
    roles.gid += shift
    edges.src += shift
    edges.dst += shift
    graph = agent.GraphQueries(roles, edges)
    assert graph.lookup_node(str(shift+1))["node"]["gid"] == str(shift+1)
    path = graph.find_path(str(shift+1), str(shift+7), 4)["route"]
    assert path["edges"][0]["dst"] == str(shift+4)
    assert graph.rank_cluster(0, 2)["nodes"][0]["gid"] == str(shift+5)


def test_required_tool_then_grounded_final_and_usage(monkeypatch, tables):
    requests = fake_api(monkeypatch, [response([call()]), response([text_message()])])
    result = agent._run_agent("fake-key", "mock-model", "Кто клиент?", 1, agent.GraphQueries(*tables))
    assert requests[0]["tool_choice"] == "required"
    assert all(body["store"] is False and body["max_output_tokens"] == 1200 for body in requests)
    tool_output = requests[1]["input"][-1]
    assert tool_output["type"] == "function_call_output" and tool_output["call_id"] == "c1"
    assert json.loads(tool_output["output"])["node"]["gid"] == "1"
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 8}
    assert result["trace"][0]["result"]["node"]["gid"] == "1"


def test_invalid_llm_arguments_become_tool_error(monkeypatch, tables):
    requests = fake_api(monkeypatch, [response([call(arguments={"gid": "999"})]), response([text_message("Данных нет.")])])
    result = agent._run_agent("fake", "mock", "Проверь", 1, agent.GraphQueries(*tables))
    assert "error" in result["trace"][0]["result"]
    assert "error" in json.loads(requests[1]["input"][-1]["output"])


def test_three_request_cap_final_disallows_tools(monkeypatch, tables):
    requests = fake_api(monkeypatch, [response([call()]), response([call(call_id="c2")]), response([text_message()])])
    agent._run_agent("fake", "mock", "Проверь", 1, agent.GraphQueries(*tables))
    assert len(requests) == 3 and requests[-1]["tool_choice"] == "none"


def test_six_call_cap_and_no_tool_bypass(monkeypatch, tables):
    fake_api(monkeypatch, [response([call(call_id=str(i)) for i in range(7)])])
    with pytest.raises(AssistantResponseError, match="6"):
        agent._run_agent("fake", "mock", "Проверь", 1, agent.GraphQueries(*tables))
    fake_api(monkeypatch, [response([text_message()])])
    with pytest.raises(AssistantResponseError, match="обязательный"):
        agent._run_agent("fake", "mock", "Проверь", 1, agent.GraphQueries(*tables))


class IdleUI:
    def __init__(self, click_local=False):
        self.session_state = {}
        self.click_local = click_local
    def expander(self, *args, **kwargs): return nullcontext()
    def write(self, *args, **kwargs): pass
    def caption(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def text_input(self, *args, **kwargs): return "1,2"
    def text_area(self, *args, **kwargs): return "Покажи клиента"
    def slider(self, *args, **kwargs): return 1
    def button(self, *args, **kwargs): return self.click_local and kwargs.get("key") == "graph_local_run"


@pytest.mark.parametrize("with_key", [False, True])
def test_render_idle_never_calls_network(monkeypatch, tables, with_key):
    monkeypatch.setattr(agent, "st", IdleUI())
    monkeypatch.setattr(agent, "read_config", lambda: {"OPENAI_API_KEY": "fake"} if with_key else {})
    monkeypatch.setattr(agent, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network call"))
    agent.render_graph_agent(*tables, 1)


def test_offline_search_works_on_click_without_api_key(monkeypatch, tables):
    ui = IdleUI(click_local=True)
    monkeypatch.setattr(agent, "st", ui)
    monkeypatch.setattr(agent, "read_config", lambda: {})
    monkeypatch.setattr(agent, "_show_result", lambda *a, **kw: None)
    monkeypatch.setattr(agent, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network call"))
    agent.render_graph_agent(*tables, 1)
    assert ui.session_state["graph_local_result"]["trace"][0]["result"]["total_matches"] == 1
