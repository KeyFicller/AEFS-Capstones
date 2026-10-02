from terminal_coding_agent.config import CHARS_PER_TOKEN_APPROX, TOOL_OUTPUT_TOKEN_LIMIT

def truncate(text: str, limit: int = TOOL_OUTPUT_TOKEN_LIMIT) -> str:
    max_chars = limit * CHARS_PER_TOKEN_APPROX
    if len(text) <= max_chars:
        return text
    marker = f"\n...[truncated, ~{limit} tokens]"
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker