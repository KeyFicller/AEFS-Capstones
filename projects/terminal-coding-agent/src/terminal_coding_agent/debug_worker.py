"""lldb worker: one live session, driven by newline-delimited JSON on stdin/stdout.

Runs under the system Python 3.9 because the lldb bindings ship as
`_lldb.cpython-39-*` while the package runs on 3.14. The package never imports
this module; it only ever launches it as a subprocess.
"""

from __future__ import annotations

import json
import sys
import threading

import lldb

_STOP_REASONS = {
    lldb.eStopReasonBreakpoint: "breakpoint",
    lldb.eStopReasonException: "exception",
    lldb.eStopReasonSignal: "signal",
    lldb.eStopReasonWatchpoint: "watchpoint",
    lldb.eStopReasonPlanComplete: "step",
}


def _stop_reason(process: lldb.SBProcess) -> str | None:
    thread = process.GetSelectedThread()
    if not thread.IsValid():
        return None
    return _STOP_REASONS.get(thread.GetStopReason())


def _stop_location(process: lldb.SBProcess) -> dict | None:
    """Where the selected frame sits, or None when the process is not stopped."""
    if not process.IsValid() or process.GetState() != lldb.eStateStopped:
        return None
    frame = process.GetSelectedThread().GetSelectedFrame()
    if not frame.IsValid():
        return None
    entry = frame.GetLineEntry()
    return {"file": entry.GetFileSpec().GetFilename(), "line": entry.GetLine()}


class Session:
    """The live debug session. Exactly one per worker process."""

    def __init__(self) -> None:
        self._debugger: lldb.SBDebugger | None = None
        self._target: lldb.SBTarget | None = None
        self._binary: str | None = None
        self._args: list = []
        self._last_stop: tuple[str, int] | None = None

    def start(self, binary: str, args: list) -> dict:
        # Same target and args: keep the session.
        if self._target is not None and binary == self._binary and list(args) == self._args:
            return {
                "ok": True,
                "output": f"target: {binary} (session kept)",
                "stop_reason": None,
            }
        self.stop()
        debugger = lldb.SBDebugger.Create()
        debugger.SetAsync(False)
        # lldb asks before killing a running process ("kill it and restart?: [Y/n]") and
        # reads the answer from stdin — which is our JSON pipe, so nobody answers and
        # HandleCommand blocks forever. The watchdog cannot help: the block is in lldb's
        # input reader, not in the process run loop. A re-`run` on a kept session hits this.
        debugger.GetCommandInterpreter().HandleCommand(
            "settings set auto-confirm true", lldb.SBCommandReturnObject()
        )
        target = debugger.CreateTarget(binary)
        if not target.IsValid():
            return {
                "ok": False,
                "output": f"could not create target: {binary}",
                "stop_reason": None,
            }
        self._debugger = debugger
        self._target = target
        self._binary = binary
        self._args = list(args)
        self._last_stop = None
        if args:
            result = lldb.SBCommandReturnObject()
            debugger.GetCommandInterpreter().HandleCommand(
                "settings set target.run-args " + " ".join(args), result, False
            )
        return {"ok": True, "output": f"target: {binary}", "stop_reason": None}

    def command(self, command: str, timeout: float) -> dict:
        if self._debugger is None:
            return {
                "ok": False,
                "output": "no session; call debug_start first",
                "stop_reason": None,
            }
        interrupted = threading.Event()
        watchdog = threading.Timer(timeout, self._interrupt, args=(interrupted,))
        watchdog.start()
        try:
            result = lldb.SBCommandReturnObject()
            self._debugger.GetCommandInterpreter().HandleCommand(command, result, False)
        finally:
            watchdog.cancel()
        output = (result.GetOutput() or "") + (result.GetError() or "")
        if interrupted.is_set():
            output += f"\n[interrupted after {timeout:g}s]"
        location = _stop_location(self._target.GetProcess())
        if location is None:
            # Running or exited: forget the last stop, so the next one is reported.
            self._last_stop = None
        elif (location["file"], location["line"]) == self._last_stop:
            # Same place as the previous report: nothing new to show.
            location = None
        else:
            self._last_stop = (location["file"], location["line"])
        return {
            "ok": bool(result.Succeeded()),
            "output": output,
            "stop_reason": _stop_reason(self._target.GetProcess()),
            "stop_location": location,
        }

    def _interrupt(self, interrupted: threading.Event) -> None:
        """Called from the watchdog thread; ``SBProcess.Stop`` is safe off the main thread."""
        process = self._target.GetProcess()
        if process.IsValid():
            process.Stop()
            interrupted.set()

    def stop(self) -> dict:
        if self._debugger is not None:
            lldb.SBDebugger.Destroy(self._debugger)
        self._debugger = None
        self._target = None
        self._binary = None
        self._args = []
        self._last_stop = None
        return {"ok": True, "output": "", "stop_reason": None}


def handle(request: dict, session: Session) -> dict:
    op = request.get("op")
    if op == "start":
        return session.start(request["binary"], list(request.get("args") or []))
    if op == "cmd":
        return session.command(str(request["command"]), float(request.get("timeout") or 30))
    if op == "stop":
        return session.stop()
    return {"ok": False, "output": f"unknown op: {op!r}", "stop_reason": None}


def _write(response: dict) -> None:
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def main() -> int:
    session = Session()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        stop = False
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"ok": False, "output": f"bad request: {exc}", "stop_reason": None}
        else:
            stop = request.get("op") == "stop"
            try:
                response = handle(request, session)
            except Exception as exc:  # noqa: BLE001 - one bad op must not kill the session
                response = {
                    "ok": False,
                    "output": f"{type(exc).__name__}: {exc}",
                    "stop_reason": None,
                }
        _write(response)
        if stop:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
