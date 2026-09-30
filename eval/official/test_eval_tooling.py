"""
Tests for the eval tooling and image caching (pytest, Gemini mocked).

1. Eval scripts default to the official eval/official/input.txt, read .txt query files,
   accept --dataset as an alias of --queries, and fail on a missing path instead of silently
   falling back to sample data.
2. generate_metrics labels results from the results.meta.json sidecar run_eval writes:
   official data is not called sample, sample data is flagged, unknown provenance is flagged.
3. An identical image re-upload reuses its vision caption: no second vision call, and the
   plan is then a semantic-cache hit.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

import eval_datasets  # noqa: E402
import generate_metrics  # noqa: E402
import main  # noqa: E402
from cache import cache_clear, cache_stats  # noqa: E402
from schema import Goal  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Dataset resolution
# ---------------------------------------------------------------------------

def test_default_dataset_is_the_official_input():
    assert not (ROOT / "queries.json").exists(), "a queries.json would take precedence; update this test"
    path, is_sample = eval_datasets.find_queries_path()
    assert path == eval_datasets.OFFICIAL_INPUT
    assert is_sample is False


def test_txt_queries_load_one_per_line():
    queries = eval_datasets.load_queries(eval_datasets.OFFICIAL_INPUT)
    assert len(queries) == 20
    assert queries[0]["id"] == "q1" and queries[0]["query"].startswith("My TechCorp A15G tablet")


def test_json_sample_still_loads_and_is_flagged():
    path, is_sample = eval_datasets.find_queries_path(str(ROOT / "queries.sample.json"))
    assert is_sample is True
    assert len(eval_datasets.load_queries(path)) == 6


def test_missing_path_is_an_error_not_a_silent_sample_fallback():
    with pytest.raises(FileNotFoundError):
        eval_datasets.find_queries_path("does/not/exist.json")


@pytest.mark.parametrize("script", ["run_eval.py", "ablation.py"])
def test_dataset_flag_is_accepted_and_missing_file_fails_loudly(script):
    r = subprocess.run(
        [sys.executable, str(ROOT / "eval" / script), "--dataset", "does/not/exist.txt"],
        capture_output=True, text=True, timeout=120,
    )
    assert "unrecognized arguments" not in r.stderr  # --dataset is a real flag
    assert r.returncode != 0
    assert "Query dataset not found: does/not/exist.txt" in r.stderr
    assert "Falling back" not in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# 2. Metrics labelling from the results.meta.json sidecar
# ---------------------------------------------------------------------------

def _label(tmp_path, queries_path=None, is_sample=False):
    results = tmp_path / "results.jsonl"
    results.write_text("{}\n")
    if queries_path is not None:
        eval_datasets.write_results_meta(results, queries_path, is_sample, n_queries=20, live=False)
    metrics = {}
    generate_metrics.apply_dataset_label(metrics, results)
    return metrics, results


def test_official_results_are_not_labelled_sample(tmp_path):
    metrics, results = _label(tmp_path, eval_datasets.OFFICIAL_INPUT, is_sample=False)
    meta = json.loads((tmp_path / "results.meta.json").read_text())
    assert meta["queries_file"] == "eval/official/input.txt"
    assert meta["catalog_file"] == "deeplinks.json" and meta["catalog_is_sample"] is False
    assert metrics["is_sample_data"] is False
    assert "eval/official/input.txt" in metrics["dataset_desc"]


def test_sample_results_are_flagged(tmp_path):
    metrics, _ = _label(tmp_path, ROOT / "queries.sample.json", is_sample=True)
    assert metrics["is_sample_data"] is True
    assert "queries.sample.json" in metrics["dataset_desc"]


def test_results_without_sidecar_are_flagged_as_unknown(tmp_path):
    metrics, _ = _label(tmp_path)
    assert metrics["is_sample_data"] is True
    assert "unrecorded dataset" in metrics["dataset_desc"]


# ---------------------------------------------------------------------------
# 3. Image re-upload reuses its caption
# ---------------------------------------------------------------------------

_GOAL = {
    "goal": "Follow these steps to perform this Touch Screen Troubleshooting",
    "title": "Touch screen responsiveness",
    "score": 0.8,
    "actions": [{"actionName": "Enable Touch Sensitivity", "description": "It will improve touch response",
                 "category": "auto", "stepGroups": [{"steps": ["Open Settings.", "Tap Display.",
                                                                 "Turn on Touch sensitivity."]}]}],
}


@pytest.fixture
def image_api(monkeypatch):
    from fastapi.testclient import TestClient

    calls = {"vision": 0, "extract": 0}

    def fake_vision(**kwargs):
        calls["vision"] += 1
        return MagicMock(text="Touch screen unresponsive in the lower half")

    def fake_extract_goals(**kwargs):
        calls["extract"] += 1
        return [Goal(**_GOAL)], {"prompt_tokens": 0, "candidates_tokens": 0}

    monkeypatch.setattr(main, "gemini_client", MagicMock(name="mock_gemini_client"))
    monkeypatch.setattr(main, "generate_content_with_failover", fake_vision)
    monkeypatch.setattr(main, "extract_goals", fake_extract_goals)
    monkeypatch.setenv("ENABLE_SELF_CRITIQUE", "false")
    cache_clear()
    yield TestClient(main.app), calls
    cache_clear()


def _upload(client, data: bytes):
    return client.post("/v1/troubleshoot-image", files={"file": ("photo.jpg", data, "image/jpeg")}).json()


def _contexts(body):
    return body["response"]["contexts"] if "response" in body else body["contexts"]


def test_identical_image_reupload_makes_no_gemini_calls(image_api):
    client, calls = image_api
    photo = b"\xff\xd8\xff" + b"same photo bytes" * 50

    first = _upload(client, photo)
    assert calls == {"vision": 1, "extract": 1}
    assert _contexts(first)[0]["title"] == "Touch screen responsiveness"
    assert first["meta"]["cache_hit"] is False

    second = _upload(client, photo)
    assert calls == {"vision": 1, "extract": 1}  # no vision call, no extraction: fully cached
    assert second["meta"]["cache_hit"] is True
    assert _contexts(second) == _contexts(first)
    assert cache_stats()["image_caption_hits"] == 1


def test_different_image_still_calls_vision(image_api):
    client, calls = image_api
    _upload(client, b"\xff\xd8\xff" + b"photo one" * 50)
    _upload(client, b"\xff\xd8\xff" + b"photo two" * 50)
    assert calls["vision"] == 2


def test_failed_vision_call_is_not_remembered(image_api, monkeypatch):
    client, calls = image_api

    def failing_vision(**kwargs):
        calls["vision"] += 1
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(main, "generate_content_with_failover", failing_vision)
    photo = b"\xff\xd8\xff" + b"photo during outage" * 50
    assert _upload(client, photo)["fallback"] == "vision_unavailable"
    assert _upload(client, photo)["fallback"] == "vision_unavailable"
    assert calls["vision"] == 2  # a transient failure is retried on the next upload, never cached
