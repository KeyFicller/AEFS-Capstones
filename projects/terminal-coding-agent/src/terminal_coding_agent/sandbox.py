"""Docker-backed execution for `run_shell`: one container per CLI session.

The graph, the model, and the file tools stay on the host. Only shell commands
cross into the container, whose only mount is the host worktree, so every write
lands where the user can inspect it after the session ends.
"""

import os
import re
import subprocess
from contextlib import suppress
from pathlib import Path

CONTAINER_WORKTREE = "/workspace"
START_TIMEOUT_SECONDS = 60
STOP_TIMEOUT_SECONDS = 30
# The container runs its own `timeout`; the outer one only guards a hung docker client.
EXEC_TIMEOUT_SLACK_SECONDS = 30
TIMEOUT_EXIT_CODE = 124

_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


class SandboxError(RuntimeError):
    """Docker could not be reached or the container could not be started."""


def container_name(session: str) -> str:
    """A Docker-legal container name derived from the session id."""
    safe = _NAME_SAFE.sub("-", session).strip("-.")
    return f"tca-{safe or 'session'}"


def require_docker(*, timeout: int = 15) -> None:
    """Fail before the REPL opens instead of on the first tool call."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxError(f"docker is not available: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "docker info failed"
        raise SandboxError(f"docker is not available: {detail}")


class DockerSandbox:
    """A long-lived `sleep infinity` container bound to one host worktree."""

    def __init__(self, *, worktree: Path, image: str, session: str) -> None:
        self.worktree = worktree.resolve()
        self.image = image
        self.name = container_name(session)

    def start(self) -> None:
        # A SIGKILLed CLI leaves `sleep infinity` running; the same session name would collide.
        self._remove_stale()
        command = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            self.name,
            "--network",
            "none",
            "-v",
            f"{self.worktree}:{CONTAINER_WORKTREE}",
            "-w",
            CONTAINER_WORKTREE,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            self.image,
            "sleep",
            "infinity",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=START_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"could not start the container: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or "docker run failed"
            raise SandboxError(f"could not start the container: {detail}")

    def run(self, command: str, timeout: int) -> tuple[int, str, str] | None:
        """Run one command in the container; None when it hit the timeout."""
        docker_command = [
            "docker",
            "exec",
            self.name,
            "timeout",
            "-k",
            "1",
            str(timeout),
            "sh",
            "-c",
            command,
        ]
        try:
            result = subprocess.run(
                docker_command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout + EXEC_TIMEOUT_SLACK_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode == TIMEOUT_EXIT_CODE:
            return None
        return result.returncode, result.stdout, result.stderr

    def close(self) -> None:
        """Stop the container; the host worktree is untouched."""
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "stop", "-t", "1", self.name],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=STOP_TIMEOUT_SECONDS,
            )

    def _remove_stale(self) -> None:
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "rm", "-f", self.name],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=STOP_TIMEOUT_SECONDS,
            )
