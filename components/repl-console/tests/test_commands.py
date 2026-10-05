from pathlib import Path

from repl_console.commands import (
    CommandContext,
    CommandOutcome,
    LocalCommand,
    PromptCommand,
    Registry,
)


class _Echo(LocalCommand):
    name = "echo"
    summary = "repeat the args"

    def run(self, ctx, args):
        return CommandOutcome(message=args)


class _Review(PromptCommand):
    name = "review"
    summary = "ask the agent to review a file"

    def run(self, ctx, args):
        return CommandOutcome(prompt=f"Review {args} and report risks.")


def _ctx() -> CommandContext:
    return CommandContext(root=Path())


def test_unknown_command_returns_none() -> None:
    assert Registry().dispatch("nope", "", _ctx()) is None


def test_a_derived_local_command_returns_a_message() -> None:
    outcome = Registry([_Echo()]).dispatch("echo", "hi there", _ctx())
    assert outcome == CommandOutcome(message="hi there")


def test_a_derived_prompt_command_returns_a_prompt() -> None:
    outcome = Registry([_Review()]).dispatch("review", "src/a.py", _ctx())
    assert outcome is not None
    assert outcome.prompt == "Review src/a.py and report risks."


def test_help_lists_every_command_in_this_registry() -> None:
    outcome = Registry([_Echo()]).dispatch("help", "", _ctx())
    assert outcome is not None
    assert outcome.message is not None
    assert "/echo" in outcome.message
    assert "/help" in outcome.message
    assert "/quit" in outcome.message


def test_quit_asks_the_repl_to_stop() -> None:
    outcome = Registry().dispatch("quit", "", _ctx())
    assert outcome is not None
    assert outcome.quit is True


def test_a_later_command_replaces_a_builtin() -> None:
    class _Quit(LocalCommand):
        name = "quit"
        summary = "custom quit"

        def run(self, ctx, args):
            return CommandOutcome(message="custom", quit=True)

    outcome = Registry([_Quit()]).dispatch("quit", "", _ctx())
    assert outcome == CommandOutcome(message="custom", quit=True)
