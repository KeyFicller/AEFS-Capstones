"""Run state under an explicit artifacts dir: the worktree itself stays clean."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from telemetry.setup import otel_jsonl_path
from terminal_coding_agent import artifacts as artifacts_module
from terminal_coding_agent.artifacts import publish_patch
from terminal_coding_agent.checkpoint import build_checkpointer, checkpoint_path
from terminal_coding_agent.ledger import trace_path
from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.summary import build_summary


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@t"),
        ("config", "user.name", "t"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    return repo


class _StubModel:
    def invoke(self, input, config=None):  # noqa: A002 - mirrors BaseChatModel
        return AIMessage(content="done")


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="hi")],
        "todo_list": [],
        "turns": 0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_rmb": 0.0,
        "stop_reason": None,
    }


def test_checkpoint_goes_to_the_artifacts_dir(tmp_path: Path) -> None:
    repo, artifacts = _repo(tmp_path), tmp_path / "state"

    build_checkpointer(repo, artifacts=artifacts)

    assert checkpoint_path(repo, artifacts).is_file()
    assert not (repo / ".agent").exists()


def test_patch_and_trace_go_to_the_artifacts_dir(tmp_path: Path) -> None:
    repo, artifacts = _repo(tmp_path), tmp_path / "state"
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    model = _StubModel()
    node = build_summary(
        AgentModels(planner=model, executor=model), worktree=repo, artifacts=artifacts
    )

    node(_state(), RunnableConfig())

    assert "-x = 1" in (artifacts / "patch.diff").read_text(encoding="utf-8")
    assert json.loads((artifacts / "trace.json").read_text(encoding="utf-8"))["stop_reason"] == (
        "completed"
    )
    assert not (repo / ".agent").exists()


def test_otel_jsonl_follows_the_artifacts_dir(tmp_path: Path) -> None:
    repo, artifacts = _repo(tmp_path), tmp_path / "state"

    assert otel_jsonl_path(repo, artifacts) == artifacts / "otel.jsonl"
    assert otel_jsonl_path(repo) == repo / ".agent" / "otel.jsonl"


def test_trace_path_defaults_to_the_worktree(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    assert trace_path(repo) == repo / ".agent" / "trace.json"
    assert trace_path(repo, tmp_path / "state") == tmp_path / "state" / "trace.json"


def test_publish_patch_writes_only_where_asked(tmp_path: Path) -> None:
    repo, artifacts = _repo(tmp_path), tmp_path / "state"

    with patch.object(artifacts_module, "HARBOR_ARTIFACTS_DIR", tmp_path / "absent"):
        written = publish_patch(repo, artifacts)

    assert written == [artifacts / "patch.diff"]
    assert not (repo / ".agent").exists()
