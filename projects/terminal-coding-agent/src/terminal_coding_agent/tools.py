"""Tools callable by the agent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

from terminal_coding_agent.config import SCRIPT_TIMEOUT_SECONDS


@tool
def create_file(file_path: str, content: str) -> str:
    """Create a file.

    Args:
        file_path: Path relative to the directory of the current script.
        content: Content of the file.

    Returns:
        str: The file was created successfully.
    """
    target_path = Path(__file__).resolve().parent / file_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return f"File {target_path} created successfully"

@tool
def read_file(file_path: str) -> str:
    """Read a file.

    Args:
        file_path: Path relative to the directory of the current script.

    Returns:
        str: The content of the file.
    """
    target_path = Path(__file__).resolve().parent / file_path
    return target_path.read_text(encoding="utf-8")

@tool
def run_script(script_path: str) -> str:
    """Run a script.

    Args:
        script_path: Path relative to the directory of the current script.

    Returns:
        str: The script ran successfully.
    """
    target_path = Path(__file__).resolve().parent / script_path
    if not target_path.exists():
        return f"File {target_path} does not exist"
    try:
        completed = subprocess.run(
            [sys.executable, str(target_path)],
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Timeout: script exceeded {SCRIPT_TIMEOUT_SECONDS}s"
    if completed.returncode == 0 and not completed.stderr:
        return completed.stdout
    return (
        f"exit_code: {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@tool
def verify_script(script_stdout: str, desired_output: str) -> str:
    """Verify the output of a script.

    Args:
        script_stdout: The stdout of the script.
        desired_output: The desired output of the script.

    Returns:
        str: "Failed" if the script output is incorrect, "Success" if the script output is correct.
    """
    if script_stdout != desired_output:
        return "Failed"
    return "Success"
