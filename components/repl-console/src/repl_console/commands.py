"""Slash-command registry. One Registry per Repl. Nothing here prints."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandOutcome:
    """Stop, print a local message, or send a prompt on as a task."""

    quit: bool = False
    message: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class CommandContext:
    root: Path
    extra_dirs: tuple[Path, ...] = ()


class Command(ABC):
    name: str
    summary: str

    @abstractmethod
    def run(self, ctx: CommandContext, args: str) -> CommandOutcome:
        """Handle one invocation. `args` is everything after the command name."""


class LocalCommand(Command):
    """Acts in the REPL itself; never reaches `on_task` unless it returns a prompt."""


class PromptCommand(Command):
    """Expands to a prompt, which the REPL sends to `on_task`."""


class HelpCommand(LocalCommand):
    name = "help"
    summary = "list the registered commands"

    def run(self, ctx: CommandContext, args: str) -> CommandOutcome:
        return CommandOutcome(message="")


class QuitCommand(LocalCommand):
    name = "quit"
    summary = "leave the session"

    def run(self, ctx: CommandContext, args: str) -> CommandOutcome:
        return CommandOutcome(quit=True)


class Registry:
    """Builtins first, then `extra`. A repeated name replaces the earlier command."""

    def __init__(self, extra: Sequence[Command] = ()) -> None:
        self._commands: dict[str, Command] = {}
        self.add(HelpCommand())
        self.add(QuitCommand())
        for command in extra:
            self.add(command)

    def add(self, command: Command) -> None:
        self._commands[command.name] = command

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._commands))

    def dispatch(self, name: str, args: str, ctx: CommandContext) -> CommandOutcome | None:
        """None means unknown. `/help` lists this registry, not a global table."""
        command = self._commands.get(name)
        if command is None:
            return None
        if isinstance(command, HelpCommand):
            lines = [
                f"/{item.name}  {item.summary}"
                for item in sorted(self._commands.values(), key=lambda item: item.name)
            ]
            return CommandOutcome(message="\n".join(lines))
        return command.run(ctx, args)
