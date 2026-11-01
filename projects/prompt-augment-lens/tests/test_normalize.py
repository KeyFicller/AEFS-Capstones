from __future__ import annotations

import json
from pathlib import Path

from prompt_augment_lens.normalize import normalize_capture_file, normalize_chat_completions
from prompt_augment_lens.paths import FIXTURES_DIR


def test_fixture_keeps_utterance_and_adds_layers() -> None:
    path = FIXTURES_DIR / "sample_agent_turn.json"
    turn = normalize_capture_file(str(path))

    assert turn.user_utterance_raw == "fix the typo in README"
    assert turn.user_utterance_in_payload == "fix the typo in README"
    assert turn.diff_summary is not None
    assert turn.diff_summary.utterance_unchanged is True
    assert turn.diff_summary.utterance_wrapped is True
    assert turn.diff_summary.bytes_model > turn.diff_summary.bytes_ui

    kinds = {layer.kind for layer in turn.layers}
    assert "system" in kinds
    assert "rules_or_instructions" in kinds
    assert "environment" in kinds
    assert "user_utterance" in kinds
    assert "tools" in kinds


def test_plain_user_message_without_wrapper() -> None:
    body = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hello"},
        ],
    }
    turn = normalize_chat_completions(body, user_utterance_raw="hello")
    assert turn.diff_summary is not None
    assert turn.diff_summary.utterance_unchanged is True
    assert turn.diff_summary.utterance_wrapped is False
    assert turn.diff_summary.bytes_model > turn.diff_summary.bytes_ui


def test_capture_roundtrip_dict_shape(tmp_path: Path) -> None:
    src = FIXTURES_DIR / "sample_agent_turn.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    out = tmp_path / "copy.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    turn = normalize_capture_file(str(out))
    data = turn.to_dict()
    assert data["vendor"] == "sample_coding_agent"
    assert data["diff_summary"]["added_layers"]
