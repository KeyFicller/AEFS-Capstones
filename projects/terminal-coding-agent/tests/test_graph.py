import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from terminal_coding_agent import graph as graph_module
from terminal_coding_agent.graph import _after_intent, make_graph


def _graph(tmp_path: Path, **configurable):
    os.environ["DEEPSEEK_API_KEY"] = "test"
    try:
        return make_graph({"configurable": {"worktree": tmp_path, **configurable}})
    finally:
        os.environ.pop("DEEPSEEK_API_KEY")


def _state(**extra):
    state = {
        "messages": [],
        "todo_list": [],
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_rmb": 0.0,
        "stop_reason": None,
        "blocked_reason": None,
    }
    state.update(extra)
    return state


def test_mermaid_png_is_skipped_without_a_path(tmp_path: Path) -> None:
    calls: list[object] = []

    def fake_png(mermaid, **kwargs):
        calls.append(mermaid)
        return b"\x89PNG\r\n\x1a\nfake"

    with patch("terminal_coding_agent.graph.draw_mermaid_png", fake_png):
        _graph(tmp_path, show_mermaid=True)

    assert calls == []


def test_mermaid_path_writes_png(tmp_path: Path, capsys) -> None:
    png_path = tmp_path / "diagrams" / "agent.png"
    seen: list[str] = []

    def fake_png(mermaid, **kwargs):
        seen.append(mermaid)
        return b"\x89PNG\r\n\x1a\nfake"

    with patch("terminal_coding_agent.graph.draw_mermaid_png", fake_png):
        _graph(tmp_path, mermaid_path=png_path)

    assert "flowchart" not in capsys.readouterr().out
    assert png_path.read_bytes().startswith(b"\x89PNG")


def test_mermaid_png_expands_the_executor_agent(tmp_path: Path) -> None:
    """The executor builds its agent at runtime, so the box must be filled in by hand."""
    seen: list[str] = []

    def fake_png(mermaid, **kwargs):
        seen.append(mermaid)
        return b"\x89PNG\r\n\x1a\nfake"

    with patch("terminal_coding_agent.graph.draw_mermaid_png", fake_png):
        _graph(tmp_path, mermaid_path=tmp_path / "agent.png")

    assert 'subgraph run_agent["run_agent"]' in seen[0]
    assert "agent__model(model)" in seen[0], "the agent's own loop must be inside the box"


def test_make_graph(tmp_path: Path) -> None:
    assert _graph(tmp_path) is not None


def test_make_graph_persists_checkpoints(tmp_path: Path) -> None:
    agent = _graph(tmp_path)

    cfg = {"configurable": {"worktree": tmp_path, "thread_id": "t1"}}
    assert agent.get_state(cfg) is not None
    assert (tmp_path / ".agent" / "checkpoints.sqlite").is_file()


def test_artifacts_dir_moves_the_checkpoint_out_of_the_worktree(tmp_path: Path) -> None:
    """`--repo` sets this so a real repository never grows a `.agent` directory."""
    artifacts = tmp_path / "state"
    agent = _graph(tmp_path, artifacts_dir=str(artifacts))

    cfg = {"configurable": {"worktree": tmp_path, "thread_id": "t1"}}
    assert agent.get_state(cfg) is not None
    assert (artifacts / "checkpoints.sqlite").is_file()
    assert not (tmp_path / ".agent").exists()


def test_invoke_without_thread_id_uses_injected_default(tmp_path: Path) -> None:
    # A checked graph raises without thread_id; with_config must supply a default.
    assert _graph(tmp_path).config["configurable"]["thread_id"]


def test_the_graph_is_one_agent_run_plus_the_stop_hook(tmp_path: Path) -> None:
    """No plan node, no per-step nodes, no approval gate: planning lives in the agent."""
    nodes = _graph(tmp_path, enable_hitl=True).builder.nodes

    assert "run_agent" in nodes
    assert "summary" in nodes
    for name in ("make_plan", "start_task", "end_task", "replan", "await_plan_approval"):
        assert name not in nodes


def _spy_on_execute_nodes(monkeypatch) -> dict:
    captured: dict = {}
    real_build = graph_module.build_execute_nodes

    def spy(models, tools, **kwargs):
        captured.update(kwargs)
        return real_build(models, tools, **kwargs)

    monkeypatch.setattr(graph_module, "build_execute_nodes", spy)
    return captured


def test_make_graph_records_a_sequence_only_when_a_path_is_configured(
    tmp_path: Path, monkeypatch
) -> None:
    """`sequence_path` is the only switch; without it the recorder must stay off."""
    captured = _spy_on_execute_nodes(monkeypatch)

    _graph(tmp_path)
    assert captured["sequence_events"] is None

    _graph(tmp_path, sequence_path=str(tmp_path / "sequence.png"))
    assert captured["sequence_events"] == []


def test_shell_anything_binds_only_run_shell(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}
    real = graph_module.make_tools

    def spy(worktree, **kwargs):
        tools = real(worktree, **kwargs)
        seen["names"] = [tool.name for tool in tools]
        seen["kwargs"] = kwargs
        return tools

    monkeypatch.setattr(graph_module, "make_tools", spy)
    captured = _spy_on_execute_nodes(monkeypatch)
    _graph(
        tmp_path,
        shell_anything=True,
        enable_web_search=True,
        enable_debug=True,
    )

    assert seen["names"] == ["run_shell"]
    assert seen["kwargs"]["shell_anything"] is True
    assert captured["shell_anything"] is True


def test_allow_dangerous_reaches_tools_and_execute_nodes(tmp_path: Path, monkeypatch) -> None:
    """The flag must survive the graph boundary: tools get it, and so do the nodes."""
    seen: dict = {}
    real = graph_module.make_tools

    def spy(worktree, **kwargs):
        seen["kwargs"] = kwargs
        return real(worktree, **kwargs)

    monkeypatch.setattr(graph_module, "make_tools", spy)
    captured = _spy_on_execute_nodes(monkeypatch)
    _graph(tmp_path, allow_dangerous=True, enable_web_search=False, enable_debug=False)

    assert seen["kwargs"]["allow_dangerous"] is True
    assert captured["allow_dangerous"] is True


def test_executor_only_reaches_the_execute_nodes(tmp_path: Path, monkeypatch) -> None:
    captured = _spy_on_execute_nodes(monkeypatch)

    _graph(tmp_path, executor_only=True)

    assert captured["executor_only"] is True


def test_make_graph_forwards_enable_hitl_to_the_execute_nodes(tmp_path: Path, monkeypatch) -> None:
    """Harbor parity: absent by default, so the middleware table cannot change."""
    captured = _spy_on_execute_nodes(monkeypatch)

    _graph(tmp_path)
    assert captured["enable_hitl"] is False

    _graph(tmp_path, enable_hitl=True)
    assert captured["enable_hitl"] is True


def test_make_graph_forwards_the_todo_renderer(tmp_path: Path, monkeypatch) -> None:
    """The panel is injected per session, so the node must receive it rather than print."""
    captured = _spy_on_execute_nodes(monkeypatch)
    render = MagicMock()

    _graph(tmp_path, todo_renderer=render)

    assert captured["todo_render"] is render


def test_after_intent_routes_chat_to_the_chat_node() -> None:
    assert _after_intent(_state(intent="chat")) == "chat"
    assert _after_intent(_state(intent="work")) == "run_agent"
    assert _after_intent(_state()) == "run_agent", "an absent verdict defaults to work"


def test_intent_gate_is_wired_only_when_enabled(tmp_path: Path) -> None:
    off = _graph(tmp_path)
    on = _graph(tmp_path, enable_intent=True)

    assert "intent" not in off.builder.nodes, "Harbor parity: node set unchanged"
    assert "chat" not in off.builder.nodes
    assert "intent" in on.builder.nodes
    assert "chat" in on.builder.nodes
