"""Tests for `eval/serve.py`. `eval/` is not a package, so load it by path."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
from tiny_llm_pipeline.generate import STAGES

SERVE = Path(__file__).resolve().parents[1] / "eval" / "serve.py"


def _load_serve():
    """Load the same file `python eval/serve.py` would run."""
    spec = importlib.util.spec_from_file_location("serve_under_test", SERVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path, tok, make_ckpt):
    """A bundle directory: `tokenizer/tokenizer.json` plus three checkpoints."""
    serve = _load_serve()
    root = tmp_path / "bundle"
    (root / "tokenizer").mkdir(parents=True)
    (root / "tokenizer" / "tokenizer.json").write_bytes(tok.path.read_bytes())
    for stage in ("pretrain", "sft", "dpo"):
        (root / stage).mkdir()
        (root / stage / "ckpt.pt").write_bytes(make_ckpt(stage).read_bytes())
    return serve, serve.Bundle.load(root, device=torch.device("cpu"))


def test_ask_contract(bundle) -> None:
    serve, loaded = bundle
    status, body = serve.handle_ask(
        loaded, {"question": "你好", "max_new": 4, "temperature": 0}
    )
    assert status == 200
    assert body["device"] == "cpu"
    assert [s["stage"] for s in body["stages"]] == list(STAGES)
    assert {s["step"] for s in body["stages"]} == {0}
    assert all(set(s["stats"]) == {"chars", "dist4", "tempo"} for s in body["stages"])
    assert all(s["stats"]["chars"] == len(s["reply"]) for s in body["stages"])
    assert body["elapsed_ms"] >= 0


def test_ask_rejects_empty_and_bad_payload(bundle) -> None:
    serve, loaded = bundle
    assert serve.handle_ask(loaded, {"question": "   "})[0] == 400
    assert serve.handle_ask(loaded, {})[0] == 400
    assert serve.handle_ask(loaded, [1, 2])[0] == 400        # not an object
    assert serve.handle_ask(loaded, {"question": 7})[0] == 400


def test_settings_clamp_out_of_range_numbers(bundle) -> None:
    serve, _ = bundle
    s = serve.settings_from({"max_new": 99_999, "temperature": -5, "rep_pen": 0.1})
    assert s.max_new == serve.MAX_NEW
    assert s.temperature == 0.0
    assert s.rep_pen == serve.REP_PEN_MIN
    # Fields the page does not expose keep their GenSettings defaults.
    assert (s.top_k, s.top_p, s.seed) == (0, 1.0, 0)
    # Unparsable values fall back to the defaults instead of raising.
    assert serve.settings_from({"max_new": "abc"}).max_new == serve.GenSettings().max_new


def test_http_round_trip_on_port_zero(bundle) -> None:
    serve, loaded = bundle
    server = serve.HTTPServer(("127.0.0.1", 0), serve.make_handler(loaded))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address
        page = urllib.request.urlopen(f"http://{host}:{port}/").read().decode("utf-8")
        assert all(stage in page for stage in STAGES)
        assert "chars=" in page                 # the JS builds the readings row
        request = urllib.request.Request(
            f"http://{host}:{port}/ask",
            data=json.dumps({"question": "你好", "max_new": 4, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(urllib.request.urlopen(request).read())
        assert [s["stage"] for s in body["stages"]] == list(STAGES)

        bad = urllib.request.Request(
            f"http://{host}:{port}/ask", data=b"not json", headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(bad)
        assert caught.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"http://{host}:{port}/nope")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
