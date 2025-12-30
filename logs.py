import os
import json
import secrets
from pathlib import Path
from datetime import datetime, timezone

from pydantic_ai.messages import ModelMessagesTypeAdapter


LOG_DIR = Path(os.getenv('LOGS_DIRECTORY', 'logs'))
LOG_DIR.mkdir(exist_ok=True)


def log_entry(agent, messages, source="user"):
    tools = []

    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())

    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "provider": agent.model.system,
        "model": agent.model.model_name,
        "tools": tools,
        "messages": dict_messages,
        "source": source
    }


def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def _extract_ts(dict_messages):
    # ищем timestamp с конца по messages и их parts
    for m in reversed(dict_messages or []):
        # 1) timestamp на уровне message
        ts = m.get("timestamp")
        if ts:
            return ts

        # 2) timestamp на уровне parts
        for p in reversed(m.get("parts") or []):
            ts = p.get("timestamp")
            if ts:
                return ts

    # 3) fallback
    return datetime.now(timezone.utc)

def _to_datetime(ts):
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        # обычно приходит ISO-строка типа "2025-12-28T04:16:58.073770+00:00"
        return datetime.fromisoformat(ts)
    return datetime.now(timezone.utc)


def log_interaction_to_file(agent, messages, source='user'):
    entry = log_entry(agent, messages, source)

    ts_raw = _extract_ts(entry.get("messages"))
    ts = _to_datetime(ts_raw)

    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return filepath
