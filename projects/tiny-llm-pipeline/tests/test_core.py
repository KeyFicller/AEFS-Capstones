from tiny_llm_pipeline.cli import main
from tiny_llm_pipeline.core import hello


def test_hello_formats_name() -> None:
    assert hello("world") == "hello, world!"


def test_cli_returns_zero(capsys) -> None:
    assert main(["world"]) == 0
    assert capsys.readouterr().out == "hello, world!\n"
