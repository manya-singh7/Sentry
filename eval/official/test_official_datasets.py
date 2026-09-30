"""
Integration tests for the official hackathon datasets (pytest, Gemini mocked).

Verifies:
1. deeplinks.json (578 entries, voiceassist:// scheme) loads through both loaders and every
   entry, including its validation deeplink, passes our Deeplink / ValidationDeepLink models.
2. All 20 queries in input.txt pass the off-domain check and run through /v1/troubleshoot
   in both RESPONSE_SHAPEs; every response validates against the official schema.py.
3. Every served deeplink is a catalog URI or voiceassist://dummy_positive; catalog links carry
   the entry's originalType and validationDeeplink, the dummy carries none.
4. The dummy's description and message are 5-7 words naming the concrete screen.
5. sample_output.json itself validates against the official schema.
"""

import json
import sys
from pathlib import Path

import pytest

OFFICIAL_DIR = Path(__file__).resolve().parent
ROOT = OFFICIAL_DIR.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(OFFICIAL_DIR))

import official_schema  # noqa: E402  (the hackathon's schema.py, unmodified)

import main  # noqa: E402
from cache import cache_clear  # noqa: E402
from schema import DUMMY_DEEPLINK_URI, Deeplink, Goal, ValidationDeepLink  # noqa: E402

QUERIES = [line.strip() for line in (OFFICIAL_DIR / "input.txt").read_text().splitlines() if line.strip()]
CATALOG = json.loads((ROOT / "deeplinks.json").read_text())["deeplinks"]
CATALOG_BY_URI = {e["deeplink"]: e for e in CATALOG}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def test_catalog_shape():
    assert len(QUERIES) == 20
    assert len(CATALOG) == 578
    assert len(CATALOG_BY_URI) == 578, "catalog URIs must be unique"
    assert all(e["deeplink"].startswith("voiceassist://") for e in CATALOG)
    assert DUMMY_DEEPLINK_URI in CATALOG_BY_URI


def test_both_loaders_read_the_official_catalog():
    from retrieval.bm25_retriever import get_retriever

    main._CATALOG_CACHE = None
    assert len(main._load_deeplink_catalog()) == 578
    retriever = get_retriever(force_reload=True)
    assert not retriever.is_sample
    # 578 minus dummy_positive and the two malformed DL-0294/0295 entries (label "onURL"/"offURL")
    assert len(retriever.indexed_docs) == 575
    assert all(d["deeplink"] != DUMMY_DEEPLINK_URI for d in retriever.indexed_docs)


def test_every_catalog_entry_passes_our_deeplink_models():
    for e in CATALOG:
        Deeplink(deeplink=e["deeplink"], description=e["description"], message=e["message"],
                 originalType=e["originalType"])
        if e.get("validation"):
            ValidationDeepLink(**e["validation"])


@pytest.mark.parametrize("action, steps, expected_id", [
    # One label, three variants: the action's wording picks on / off / view.
    ("Turn On Bluetooth", ["Open Settings.", "Tap Connections.", "Turn on Bluetooth."], "DL-0495"),
    ("Disable Bluetooth", ["Open Settings.", "Tap Connections.", "Turn off Bluetooth."], "DL-0494"),
    ("Open Bluetooth Settings", ["Open Settings.", "Tap Connections.", "Tap Bluetooth."], "DL-0044"),
    # Exact label beats a longer lookalike ("Bluetooth scanning") and a misleading message.
    ("Adjust Screen Timeout", ["Open Settings.", "Tap Display.", "Tap Screen timeout."], "DL-0220"),
    # Parenthetical in the label is ignored: "Back up data (TechCorp Cloud)".
    ("Back Up Phone Data", ["Open Settings.", "Tap on Accounts and backup.", "Select Back up data."], "DL-0542"),
])
def test_label_match_picks_the_named_setting(action, steps, expected_id):
    from retrieval import match_by_label

    entry, score = match_by_label(CATALOG, action, "", steps)
    assert entry["id"] == expected_id
    assert score == 1.0


def test_label_match_ignores_path_steps_and_junk_entries():
    from retrieval import get_retriever, match_by_label

    # "Tap Accessibility." is the path to Assistant menu, not the destination.
    assert match_by_label(CATALOG, "Disable Assistant Menu", "", [
        "Open Settings.", "Tap Accessibility.", "Tap Interaction and dexterity.", "Turn off Assistant menu."]) is None
    # DL-0294/0295 have "onURL"/"offURL" as their label: never matched, never indexed.
    retriever = get_retriever(force_reload=True)
    assert not {"DL-0294", "DL-0295"} & {d["id"] for d in retriever.indexed_docs}


@pytest.mark.parametrize("description, expected", [
    # The official sample's own descriptions are 9 and 12 words; our schema allows 5-7.
    ("It will facilitate secure data transfer between your devices", "It will facilitate secure data transfer"),
    ("It will help you locate the nearest TechCorp service center and schedule",
     "It will help locate nearest TechCorp service"),
    ("It will turn off the adaptive brightness feature", "It will turn off adaptive brightness feature"),
])
def test_long_descriptions_are_shortened_without_dangling_words(description, expected):
    from schema import Action

    short = main._shorten_description(description)
    assert short == expected
    assert 5 <= len(short.split()) <= 7
    assert short.split()[-1].lower() not in main._DANGLING_END
    Action(actionName="Test Action", description=short, category="manual", stepGroups=[{"steps": ["Step."]}])


def test_sample_output_validates_against_official_schema():
    sample = json.loads((OFFICIAL_DIR / "sample_output.json").read_text())
    official_schema.ContextDeeplinkResponse(**sample["response"])


# ---------------------------------------------------------------------------
# End-to-end over input.txt (Gemini mocked)
# ---------------------------------------------------------------------------

def _mock_goal() -> Goal:
    """A display plan with one auto action per link rule, plus manual and critical actions."""
    return Goal(**{
        "goal": "Follow these steps to perform this Screen Display Troubleshooting",
        "title": "Screen display issue",
        "score": 0.8,
        "actions": [
            {"actionName": "Adjust Screen Brightness", "description": "It will change display brightness level",
             "category": "auto", "stepGroups": [{"steps": ["Open Settings.", "Tap Display.", "Drag the Brightness slider."]}]},
            {"actionName": "Configure Game Booster", "description": "It will open game booster settings",
             "category": "auto", "stepGroups": [{"steps": ["Open Settings.", "Tap Advanced features.", "Tap Game Booster."]}]},
            {"actionName": "Visit Authorized Service Center", "description": "It will get the screen inspected",
             "category": "manual", "stepGroups": [{"steps": ["Contact an authorized TechCorp Service Center."]}]},
            {"actionName": "Restart Device in Safe Mode", "description": "It will disable all third party apps",
             "category": "critical", "stepGroups": [{"steps": ["Press and hold the Side key.", "Tap Safe mode."]}]},
        ],
    })


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    calls = []

    def fake_extract_goals(**kwargs):
        calls.append(kwargs["query"])
        return [_mock_goal()], {"prompt_tokens": 0, "candidates_tokens": 0}

    monkeypatch.setattr(main, "extract_goals", fake_extract_goals)
    monkeypatch.setenv("ENABLE_SELF_CRITIQUE", "false")
    monkeypatch.setenv("ENABLE_QUERY_VARIATIONS", "false")
    cache_clear()
    yield TestClient(main.app), calls
    cache_clear()


def _contexts(body: dict) -> list:
    return body["response"]["contexts"] if "response" in body else body["contexts"]


def _check_links(contexts: list) -> None:
    for goal in contexts:
        for action in goal["actions"]:
            for sg in action["stepGroups"]:
                link, validation = sg.get("actionableDeeplink"), sg.get("validationDeeplink")
                if action["category"] == "manual" or link is None:
                    assert link is None and validation is None
                    continue
                uri = link["deeplink"]
                if uri == DUMMY_DEEPLINK_URI:
                    assert validation is None
                    assert 5 <= len(link["description"].split()) <= 7
                    assert 5 <= len(link["message"].split()) <= 7
                    continue
                entry = CATALOG_BY_URI[uri]  # KeyError == uncatalogued link served
                assert link["originalType"] == entry["originalType"]
                if entry.get("validation"):
                    assert validation["deeplink"] == entry["validation"]["deeplink"]
                    assert validation["key"] == entry["validation"]["key"]
                else:
                    assert validation is None


@pytest.mark.parametrize("query", QUERIES, ids=[f"q{i + 1:02d}" for i in range(len(QUERIES))])
def test_input_query_passes_off_domain_check(query):
    assert main.has_device_keywords(query), query


@pytest.mark.parametrize("shape", ["flat", "appendix_b"])
def test_all_input_queries_end_to_end(client, monkeypatch, shape):
    http, calls = client
    monkeypatch.setattr(main, "RESPONSE_SHAPE", shape)

    for query in QUERIES:
        r = http.post("/v1/troubleshoot", json={"query": query})
        assert r.status_code == 200, (query, r.text[:300])
        body = r.json()
        contexts = _contexts(body)
        assert contexts, (query, body.get("fallback"))

        # The official schema is the contract judges validate against.
        official_schema.ContextDeeplinkResponse(contexts=contexts)
        if shape == "appendix_b":
            assert body["query"] == query  # same top-level layout as sample_output.json
        _check_links(contexts)

    assert len(calls) == len(QUERIES)  # every query reached the (mocked) model, none rejected early


def test_link_rules_on_the_mock_plan(client):
    http, _ = client
    actions = _contexts(http.post("/v1/troubleshoot", json={"query": QUERIES[0]}).json())[0]["actions"]
    by_name = {a["actionName"]: a["stepGroups"][0] for a in actions}

    brightness = by_name["Adjust Screen Brightness"]["actionableDeeplink"]
    assert brightness["deeplink"] in CATALOG_BY_URI and brightness["deeplink"] != DUMMY_DEEPLINK_URI

    booster = by_name["Configure Game Booster"]["actionableDeeplink"]
    assert booster["deeplink"] == DUMMY_DEEPLINK_URI
    assert booster["description"] == "Opens the Game Booster settings screen"

    assert by_name["Visit Authorized Service Center"]["actionableDeeplink"] is None
    assert by_name["Restart Device in Safe Mode"]["actionableDeeplink"] is None
