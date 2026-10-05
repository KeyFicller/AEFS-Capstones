from pathlib import Path

import pytest
from repl_console.repl import Repl


def test_shift_tab_runs_the_binding_and_the_toolbar_is_accepted(tmp_path: Path) -> None:
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.output import DummyOutput

    seen: list[str] = []
    flips: list[str] = []
    bindings = KeyBindings()

    @bindings.add("s-tab")
    def _flip(event) -> None:
        flips.append("flip")
        event.app.invalidate()

    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[Zhi\r/quit\r")
        Repl(
            title="t",
            info={},
            root=tmp_path,
            on_task=lambda message: seen.append(str(message.content)),
            toolbar=lambda: [("bold", "mode")],
            key_bindings=bindings,
            stdin=pipe,
            output=DummyOutput(),
        ).run()

    assert flips == ["flip"]
    assert seen == ["hi"]


def test_tab_completes_a_slash_command(tmp_path: Path) -> None:
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document
    from repl_console.repl import _Completer

    completer = _Completer(roots=(tmp_path,), command_names=("help", "quit"))
    document = Document("/he", 3)
    texts = [item.text for item in completer.get_completions(document, CompleteEvent())]
    assert texts == ["/help"]
