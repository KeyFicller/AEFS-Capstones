from __future__ import annotations

from prompt_augment_lens.proxy import infer_utterance


def test_infer_utterance_from_user_query_tag() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "<user_query>\nhello there\n</user_query>"},
        ]
    }
    assert infer_utterance(body) == "hello there"
