"""File-based runtime control plane shared by bot and dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

CONTROL_STATE_PATH = Path("data/runtime_control.json")
COMMAND_QUEUE_PATH = Path("data/runtime_commands.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict:
    return {
        "paused": False,
        "overrides": {},
        "updated_at": _now_iso(),
    }


def get_control_state() -> dict:
    if not CONTROL_STATE_PATH.exists():
        return _default_state()

    try:
        data = json.loads(CONTROL_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()

    state = _default_state()
    if isinstance(data, dict):
        state["paused"] = bool(data.get("paused", state["paused"]))
        overrides = data.get("overrides", {})
        state["overrides"] = dict(overrides) if isinstance(overrides, dict) else {}
        state["updated_at"] = str(data.get("updated_at", state["updated_at"]))
    return state


def update_control_state(*, paused: bool | None = None, overrides: dict | None = None) -> dict:
    state = get_control_state()

    if paused is not None:
        state["paused"] = bool(paused)

    if overrides is not None:
        merged = dict(state.get("overrides", {}))
        for key, value in dict(overrides).items():
            if value is None:
                merged.pop(str(key), None)
            else:
                merged[str(key)] = value
        state["overrides"] = merged

    state["updated_at"] = _now_iso()

    CONTROL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def set_paused(paused: bool) -> dict:
    return update_control_state(paused=paused)


def enqueue_command(action: str, payload: dict | None = None) -> dict:
    command = {
        "id": uuid4().hex,
        "action": str(action),
        "payload": dict(payload or {}),
        "created_at": _now_iso(),
    }

    COMMAND_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMMAND_QUEUE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(command, ensure_ascii=True) + "\n")
    return command


def drain_commands() -> list[dict]:
    if not COMMAND_QUEUE_PATH.exists():
        return []

    commands: list[dict] = []
    try:
        lines = COMMAND_QUEUE_PATH.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                commands.append(parsed)
        COMMAND_QUEUE_PATH.write_text("", encoding="utf-8")
    except Exception:
        pass

    return commands
