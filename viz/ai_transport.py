"""OpenAI Responses transport with locally validated, evidence-linked output."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def read_config():
    """Read only the two supported settings; environment takes precedence."""
    config = {}
    try:
        lines = (Path(__file__).resolve().parents[1] / ".env").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        lines = []
    for line in lines:
        name, separator, value = line.partition("=")
        name = name.strip()
        if separator and name in {"OPENAI_API_KEY", "OPENAI_MODEL"}:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            config[name] = value
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if name in os.environ:
            config[name] = os.environ[name].strip()
    return config


def error_message(status):
    """Safe HTTP error messages: never include provider response bodies."""
    messages = {
        401: "OpenAI API: ключ не принят (HTTP 401). Проверьте OPENAI_API_KEY.",
        403: "OpenAI API: доступ запрещён (HTTP 403). Проверьте разрешения проекта.",
        404: "OpenAI API: модель недоступна (HTTP 404). Проверьте OPENAI_MODEL и доступ к модели.",
        429: "OpenAI API: превышен лимит запросов или бюджета (HTTP 429). Проверьте лимиты API и повторите позже.",
    }
    if status in messages:
        return messages[status]
    if isinstance(status, int) and 500 <= status <= 599:
        return f"OpenAI API временно недоступен (HTTP {status}). Повторите позже."
    if isinstance(status, int):
        return f"OpenAI API отклонил запрос (HTTP {status}). Проверьте настройки модели и формат запроса."
    return "OpenAI API отклонил запрос. Проверьте настройки модели."


def _fact_ids(context):
    facts = context.get("facts") if isinstance(context, dict) else None
    if not isinstance(facts, list) or not facts:
        raise ValueError("В контексте отсутствуют факты для обоснования ответа.")
    identifiers = []
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("Некорректный формат фактов в контексте.")
        identifier, text = fact.get("id"), fact.get("text")
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 128:
            raise ValueError("Некорректный идентификатор факта в контексте.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("В контексте есть факт без текста.")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Идентификаторы фактов в контексте повторяются.")
    return identifiers


def _validate_answer(answer, fact_ids):
    fields = {"explanation", "evidence_ids", "next_steps", "insufficient_context"}
    if not isinstance(answer, dict) or set(answer) != fields:
        raise ValueError("Ответ помощника не соответствует ожидаемой структуре.")
    explanation = answer["explanation"]
    if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 900:
        raise ValueError("Помощник вернул пустое или слишком длинное объяснение.")
    if type(answer["insufficient_context"]) is not bool:
        raise ValueError("Ответ помощника содержит неверный признак достаточности данных.")
    evidence = answer["evidence_ids"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 64:
        raise ValueError("Ответ должен ссылаться хотя бы на один факт из контекста.")
    if any(not isinstance(identifier, str) or identifier not in fact_ids for identifier in evidence):
        raise ValueError("Ответ ссылается на факт, которого нет в переданных данных.")
    if len(set(evidence)) != len(evidence):
        raise ValueError("Ответ содержит повторяющиеся ссылки на факты.")
    steps = answer["next_steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 3:
        raise ValueError("Помощник вернул неверный список следующих шагов.")
    if any(not isinstance(step, str) or not step.strip() or len(step) > 600 for step in steps):
        raise ValueError("Помощник вернул пустой или слишком длинный следующий шаг.")
    if any(char.isdigit() for text in [explanation, *steps] for char in text):
        raise ValueError("Числа должны отображаться из исходных фактов, а не генерироваться в объяснении.")
    return answer


def request_explanation(api_key, model, question, context, instructions):
    """Return validated answer/usage; propagate HTTP/network errors to the UI.

    Schema validation checks structure and citation membership, not whether
    arbitrary natural-language statements are entailed by the cited facts.
    """
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("Вопрос должен содержать от 1 до 2000 символов.")
    fact_ids = _fact_ids(context)
    schema = {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string", "enum": fact_ids}},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "insufficient_context": {"type": "boolean"},
        },
        "required": ["explanation", "evidence_ids", "next_steps", "insufficient_context"],
        "additionalProperties": False,
    }
    body = {
        "model": model,
        "instructions": instructions + (
            "\nВерни объект по заданной JSON-схеме. explanation: непустое объяснение до 900 символов; "
            "evidence_ids: от 1 до 64 разных id фактов из context.facts, на которых основан ответ; "
            "next_steps: от 1 до 3 непустых шагов по 600 символов максимум; "
            "insufficient_context: true, если для ответа на вопрос данных недостаточно. "
            "Ссылки на факты должны подтверждать сказанное, не отрицай наблюдаемые переводы."
        ),
        "store": False,
        "max_output_tokens": 1600,
        "text": {"format": {"type": "json_schema", "name": "node_explanation", "strict": True, "schema": schema}},
    }
    try:
        body["input"] = json.dumps({"question": question.strip(), "context": context}, ensure_ascii=False, allow_nan=False)
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Контекст не удалось подготовить к отправке.") from None
    request = Request(
        "https://api.openai.com/v1/responses", data=data, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        try:
            payload = json.load(response)
        except (ValueError, UnicodeError):
            raise ValueError("OpenAI API вернул некорректный JSON.") from None
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise ValueError("Помощник не завершил ответ. Уточните вопрос и повторите запрос.")
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("OpenAI API не вернул содержимое ответа.")
    texts = []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("OpenAI API вернул некорректную структуру ответа.")
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            raise ValueError("OpenAI API вернул сообщение без содержимого.")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("OpenAI API вернул некорректную структуру сообщения.")
            if part.get("type") == "refusal":
                raise ValueError("Модель отказалась формировать объяснение. Попробуйте уточнить вопрос.")
            if part.get("type") == "output_text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError("OpenAI API вернул неверный текстовый ответ.")
                texts.append(text)
    if not texts or not "".join(texts).strip():
        raise ValueError("Помощник не вернул текстовое объяснение.")
    try:
        answer = json.loads("".join(texts))
    except ValueError:
        raise ValueError("Объяснение помощника не удалось прочитать как JSON.") from None
    answer = _validate_answer(answer, set(fact_ids))
    raw_usage = payload.get("usage")
    usage = {}
    if isinstance(raw_usage, dict):
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = raw_usage.get(name)
            if type(value) is int and value >= 0:
                usage[name] = value
    return {"answer": answer, "usage": usage}
