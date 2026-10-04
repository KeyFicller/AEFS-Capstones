"""Slash-command registry: derive + register is the extension point."""

from pathlib import Path

import pytest
from terminal_coding_agent import commands


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(commands, "COMMANDS", {})


class _Echo(commands.LocalCommand):
    name = "echo"
    summary = "repeat the args"

    def run(self, ctx, args):
        return commands.CommandOutcome(message=args)


class _Review(commands.PromptCommand):
    name = "review"
    summary = "ask the agent to review a file"

    def run(self, ctx, args):
        return commands.CommandOutcome(prompt=f"Review {args} and report risks.")


def _register_builtins() -> None:
    for command in commands.builtin_commands():
        commands.register(command)


def test_unknown_command_returns_none() -> None:
    assert commands.dispatch("nope", "", commands.CommandContext(worktree=Path())) is None


def test_a_derived_local_command_returns_a_message() -> None:
    commands.register(_Echo())
    outcome = commands.dispatch("echo", "hi there", commands.CommandContext(worktree=Path()))
    assert outcome == commands.CommandOutcome(message="hi there")


def test_a_derived_prompt_command_returns_a_prompt() -> None:
    commands.register(_Review())
    outcome = commands.dispatch("review", "src/a.py", commands.CommandContext(worktree=Path()))
    assert outcome.prompt == "Review src/a.py and report risks."


def test_help_lists_every_registered_command() -> None:
    _register_builtins()
    commands.register(_Echo())
    outcome = commands.dispatch("help", "", commands.CommandContext(worktree=Path()))
    assert "/echo" in outcome.message
    assert "/help" in outcome.message
    assert "/quit" in outcome.message


def test_quit_asks_the_repl_to_stop() -> None:
    _register_builtins()
    outcome = commands.dispatch("quit", "", commands.CommandContext(worktree=Path()))
    assert outcome.quit is True
