import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"


def test_vendor_script_copies_budget_and_telemetry_only() -> None:
    script = ROOT / "scripts" / "vendor_components.sh"
    subprocess.run(["bash", str(script)], check=True)
    assert (VENDOR / "budget" / "pyproject.toml").is_file()
    assert (VENDOR / "telemetry" / "src" / "telemetry" / "__init__.py").is_file()
    assert not (VENDOR / "repl-console").exists()
    assert not (VENDOR / "budget" / ".pytest_cache").exists()
