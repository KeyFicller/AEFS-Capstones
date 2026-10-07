"""Clock tool: the model has no other way to learn the current instant.

No `zoneinfo` here: the stated scope is "now", and a local offset from
`astimezone()` needs no tzdata in the image.
"""

from datetime import UTC, datetime

from langchain_core.tools import BaseTool, tool


def _now() -> datetime:
    """The test seam: tests inject a fixed instant instead of reading the wall clock."""
    return datetime.now(UTC)


def build_current_time() -> BaseTool:
    @tool
    def current_time() -> str:
        """Report the current UTC time, the current local time, and the Unix epoch.

        Returns:
            Three lines: utc, local (with its UTC offset), and epoch in seconds.
        """
        moment = _now()
        utc = moment.astimezone(UTC)
        local = moment.astimezone()
        return (
            f"utc:   {utc.isoformat()}  {utc.strftime('%A')}\n"
            f"local: {local.isoformat()}  {local.strftime('%A')}\n"
            f"epoch: {int(moment.timestamp())}"
        )

    return current_time
