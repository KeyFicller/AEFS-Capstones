from __future__ import annotations

from prompt_augment_lens.normalize import normalize_chat_completions


def test_anthropic_top_level_system_and_tools() -> None:
    body = {
        "model": "claude-demo",
        "system": "You are Claude Code.",
        "messages": [
            {"role": "user", "content": "say hi"},
        ],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            }
        ],
    }
    turn = normalize_chat_completions(
        body,
        user_utterance_raw="say hi",
        vendor="claude_code",
    )
    assert turn.diff_summary is not None
    assert turn.diff_summary.bytes_model > turn.diff_summary.bytes_ui
    kinds = {layer.kind for layer in turn.layers}
    assert "system" in kinds
    assert "tools" in kinds
