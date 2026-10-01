"""Text statistics."""


def count_all(text: str) -> dict[str, int]:
    """Line, word and character counts for ``text``."""
    return {
        "line": len(text.splitlines()),
        "words": len(text.split()),
        "chars": len(text),
    }
