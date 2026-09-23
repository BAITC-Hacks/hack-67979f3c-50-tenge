"""Transport validation through a mocked urlopen; no live key is read."""
import io
import json
from urllib.error import HTTPError

import pytest

from viz import ai_transport as transport


CONTEXT = {"facts": [{"id": "outgoing", "text": "Наблюдается 1 исходящий перевод."}]}


def answer():
    return {"explanation": "Наблюдается 1 исходящий перевод.", "evidence_ids": ["outgoing"],
            "next_steps": ["Проверить период за пределами выборки."], "insufficient_context": False}


def payload(value=None):
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(answer() if value is None else value)}]}],
        "usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}}


def fake_api(monkeypatch, value):
    calls = []
    def fake(request, timeout):
        assert timeout == 30
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps(value).encode())
    monkeypatch.setattr(transport, "urlopen", fake)
    return calls


def invoke():
    return transport.request_explanation("fake-key", "mock-model", "Что наблюдается?", CONTEXT, "Только факты.")


def test_strict_schema_request_and_valid_result(monkeypatch):
    calls = fake_api(monkeypatch, payload())
    result = invoke()
    assert result["answer"] == answer()
    assert result["usage"]["total_tokens"] == 40
    body = calls[0]
    assert body["store"] is False and body["max_output_tokens"] == 1600
    format_ = body["text"]["format"]
    assert format_["type"] == "json_schema" and format_["strict"] is True
    assert format_["schema"]["additionalProperties"] is False
    assert format_["schema"]["properties"]["evidence_ids"]["items"]["enum"] == ["outgoing"]
    assert json.loads(body["input"])["context"] == CONTEXT


@pytest.mark.parametrize("field,value", [
    ("evidence_ids", ["fabricated"]), ("evidence_ids", []),
    ("explanation", ""), ("explanation", "x"*901),
    ("insufficient_context", 0), ("next_steps", []), ("next_steps", ["x"]*4),
    ("next_steps", ["x"*601]), ("explanation", {"not": "text"}),
])
def test_invalid_answer_rejected_locally(monkeypatch, field, value):
    invalid = answer(); invalid[field] = value
    fake_api(monkeypatch, payload(invalid))
    with pytest.raises(transport.AssistantResponseError): invoke()


@pytest.mark.parametrize("value", [
    {"status": "incomplete", "output": []},
    {"status": "completed", "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "provider secret"}]}]},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "provider secret invalid json"}]}]},
    {"status": "completed", "output": [None]},
    {"status": "completed", "output": [{"type": "message", "content": None}]},
])
def test_refusal_incomplete_malformed_safe_errors(monkeypatch, value):
    fake_api(monkeypatch, value)
    with pytest.raises(transport.AssistantResponseError) as exc: invoke()
    assert "provider secret" not in str(exc.value)


def test_invalid_question_or_fact_fails_without_network(monkeypatch):
    monkeypatch.setattr(transport, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network call"))
    for question, context in [("", CONTEXT), ("x"*2001, CONTEXT), ("Q", {"facts": []}),
                              ("Q", {"facts": CONTEXT["facts"]*2})]:
        with pytest.raises(transport.AssistantResponseError):
            transport.request_explanation("fake", "mock", question, context, "instructions")


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503, 400])
def test_http_status_has_safe_actionable_message(monkeypatch, status):
    error = HTTPError("https://api.openai.com/v1/responses", status, "provider secret", {}, io.BytesIO(b"secret-body"))
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(transport, "urlopen", fail)
    with pytest.raises(HTTPError) as exc: invoke()
    message = transport.error_message(exc.value.code)
    assert str(status) in message and "secret" not in message


def test_config_reads_only_settings_and_environment_wins(monkeypatch, tmp_path):
    module = tmp_path / "viz" / "ai_transport.py"
    module.parent.mkdir()
    (tmp_path / ".env").write_text("OPENAI_API_KEY='file-fake'\nOPENAI_MODEL=\"file-model\"\nOTHER=ignore\n", encoding="utf-8")
    monkeypatch.setattr(transport, "__file__", str(module))
    monkeypatch.setenv("OPENAI_API_KEY", "env-fake")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert transport.read_config() == {"OPENAI_API_KEY": "env-fake", "OPENAI_MODEL": "file-model"}
