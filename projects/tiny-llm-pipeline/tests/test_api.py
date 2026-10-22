"""Tests for `eval/api.py`. `eval/` is not a package, so load it by path."""

import importlib.util
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from tiny_llm_pipeline.generate import STAGES

API = Path(__file__).resolve().parents[1] / "eval" / "api.py"


@pytest.fixture
def client(tmp_path, tok, make_ckpt):
    """The app with a CPU bundle of three tiny checkpoints injected."""
    spec = importlib.util.spec_from_file_location("api_under_test", API)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "bundle"
    (root / "tokenizer").mkdir(parents=True)
    (root / "tokenizer" / "tokenizer.json").write_bytes(tok.path.read_bytes())
    for stage in STAGES:
        (root / stage).mkdir()
        (root / stage / "ckpt.pt").write_bytes(make_ckpt(stage).read_bytes())
    bundle = module.Bundle.load(root, device=torch.device("cpu"))
    module.app.dependency_overrides[module.get_bundle] = lambda: bundle
    return TestClient(module.app)


def test_one_post_returns_every_stage(client) -> None:
    response = client.post("/ask", json={"question": "你好", "max_new": 4, "temperature": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "cpu"
    assert body["elapsed_ms"] >= 0
    assert [s["stage"] for s in body["stages"]] == list(STAGES)
    assert {s["step"] for s in body["stages"]} == {0}
    assert all(set(s["stats"]) == {"chars", "dist4", "tempo"} for s in body["stages"])


def test_blank_question_is_400_and_bad_knobs_are_422(client) -> None:
    assert client.post("/ask", json={"question": "   "}).status_code == 400
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "你好", "max_new": 0}).status_code == 422
    assert client.post("/ask", json={"question": "你好", "rep_pen": 3}).status_code == 422


def test_bad_knobs_are_422_in_the_page_s_error_shape(client) -> None:
    response = client.post("/ask", json={"question": "你好", "temperature": 3.0})
    assert response.status_code == 422
    assert "temperature" in response.json()["error"]


def test_seed_is_fresh_by_default_and_replayable_when_pinned(client) -> None:
    body = {"question": "你好", "max_new": 4, "temperature": 1.0}
    first = client.post("/ask", json=body).json()
    second = client.post("/ask", json=body).json()
    assert first["seed"] != second["seed"]  # a second click resamples

    pinned = {**body, "seed": first["seed"]}
    replay = client.post("/ask", json=pinned).json()
    assert replay["seed"] == first["seed"]
    assert [s["reply"] for s in replay["stages"]] == [s["reply"] for s in first["stages"]]


def test_openapi_lists_the_endpoint(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/ask" in schema["paths"]


def test_root_serves_the_form(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert all(stage in response.text for stage in STAGES)
    # The page's default must match `GenSettings`, or the page silently runs a
    # different budget than `eval` and `serve.py`'s CLI.
    assert 'id="max_new" type="number" step="1" min="1" max="512" value="128"' in response.text
