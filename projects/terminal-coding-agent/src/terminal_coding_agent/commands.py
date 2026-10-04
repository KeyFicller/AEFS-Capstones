"""Slash-command registry. New commands derive from `LocalCommand` / `PromptCommand`
and call `register`; nothing here prints — `message` goes back to `ui` to render.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandOutcome:
    """What the REPL should do: stop, print a local message, or send a prompt to the graph."""

    quit: bool = False
    message: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class CommandContext:
    worktree: Path


class Command(ABC):
    name: str
    summary: str

    @abstractmethod
    def run(self, ctx: CommandContext, args: str) -> CommandOutcome:
        """Handle one invocation; `args` is everything after the command name."""


class LocalCommand(Command):
    """Acts in the REPL itself; never reaches the graph."""


class PromptCommand(Command):
    """Expands to a prompt, which the REPL sends to the graph as a task."""


class HelpCommand(LocalCommand):
    name = "help"
    summary = "list the registered commands"

    def run(self, ctx: CommandContext, args: str) -> CommandOutcome:
        lines = [
            f"/{command.name}  {command.summary}"
            for command in sorted(COMMANDS.values(), key=lambda item: item.name)
        ]
        return CommandOutcome(message="\n".join(lines))


class QuitCommand(LocalCommand):
    name = "quit"
    summary = "leave the session"

    def run(self, ctx: CommandContext, args: str) -> CommandOutcome:
        return CommandOutcome(quit=True)


COMMANDS: dict[str, Command] = {}


def register(command: Command) -> None:
    COMMANDS[command.name] = command


def builtin_commands() -> list[Command]:
    return [HelpCommand(), QuitCommand()]


def dispatch(name: str, args: str, ctx: CommandContext) -> CommandOutcome | None:
    """None means unknown: the REPL reports it and does not call the graph."""
    command = COMMANDS.get(name)
    if command is None:
        return None
    return command.run(ctx, args)
