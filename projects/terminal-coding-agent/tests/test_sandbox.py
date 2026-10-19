import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from terminal_coding_agent import sandbox as sandbox_module
from terminal_coding_agent.sandbox import (
    CONTAINER_WORKTREE,
    DockerSandbox,
    SandboxError,
    container_name,
    require_docker,
)


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_container_name_is_docker_legal() -> None:
    assert container_name("cli/1 2") == "tca-cli-1-2"
    assert container_name("") == "tca-session"


def test_start_mounts_only_the_worktree(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        calls.append(command)
        return _result(stdout="container-id")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    box = DockerSandbox(worktree=tmp_path, image="img:1", session="s1")
    box.start()

    assert calls[0] == ["docker", "rm", "-f", "tca-s1"]
    run = calls[1]
    assert run[:3] == ["docker", "run", "-d"]
    assert "--rm" in run and "--name" in run and "tca-s1" in run
    assert "--network" in run and run[run.index("--network") + 1] == "none"
    assert run[run.index("-v") + 1] == f"{tmp_path.resolve()}:{CONTAINER_WORKTREE}"
    assert run[run.index("-w") + 1] == CONTAINER_WORKTREE
    assert run[run.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert run[-3:] == ["img:1", "sleep", "infinity"]


def test_start_reports_a_failed_container(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ARG001
        return _result(returncode=1, stderr="no such image")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    box = DockerSandbox(worktree=tmp_path, image="missing", session="s1")

    with pytest.raises(SandboxError, match="no such image"):
        box.start()


def test_run_wraps_timeout_inside_the_container(tmp_path: Path, monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_run(command, **kwargs):
        seen.append(command)
        assert kwargs["stdin"] is subprocess.DEVNULL
        return _result(stdout="hi\n")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    box = DockerSandbox(worktree=tmp_path, image="img", session="s1")

    assert box.run("echo hi", 7) == (0, "hi\n", "")
    assert seen[0][:3] == ["docker", "exec", "tca-s1"]
    assert seen[0][3:7] == ["timeout", "-k", "1", "7"]
    assert seen[0][-2:] == ["-c", "echo hi"]


def test_run_maps_timeout_to_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module.subprocess, "run", lambda *a, **k: _result(124))
    box = DockerSandbox(worktree=tmp_path, image="img", session="s1")

    assert box.run("sleep 99", 1) is None


def test_run_maps_a_hung_client_to_none(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ARG001
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    box = DockerSandbox(worktree=tmp_path, image="img", session="s1")

    assert box.run("sleep 99", 1) is None


def test_close_stops_the_container(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        calls.append(command)
        return _result()

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    DockerSandbox(worktree=tmp_path, image="img", session="s1").close()

    assert calls[0][:2] == ["docker", "stop"]


def test_require_docker_reports_a_missing_daemon(monkeypatch) -> None:
    monkeypatch.setattr(
        sandbox_module.subprocess,
        "run",
        lambda *a, **k: _result(returncode=1, stderr="Cannot connect"),
    )

    with pytest.raises(SandboxError, match="Cannot connect"):
        require_docker()


def test_require_docker_reports_a_missing_binary(monkeypatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ARG001
        raise FileNotFoundError("docker")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)

    with pytest.raises(SandboxError, match="not available"):
        require_docker()
