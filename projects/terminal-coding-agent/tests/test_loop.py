from terminal_coding_agent.loop import MAX_TURNS, run


def test_run_completes_and_calls_tool() -> None:
    result = run("say hello")
    assert result.done is True
    assert result.plan == ["say hello"]
    assert result.observations == ["hello, world!"]


def test_run_stops_at_turn_limit() -> None:
    result = run("never converge", loop_forever=True)
    assert result.done is False
    assert result.turns == MAX_TURNS
