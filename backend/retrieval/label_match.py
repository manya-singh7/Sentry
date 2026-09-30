"""
Exact Settings-label matching over the official deeplinks.json.

Every catalog entry's validation.key is the literal label of its Settings screen or toggle
("Screen timeout", "Navigation bar", "Mobile data"). When an action names that label, in its
own name or in a step ("Turn on Mobile data."), that is a far stronger signal than bag-of-words
overlap, which confuses e.g. "Bluetooth" with "Bluetooth scanning" or "Screen timeout" with a
time-zone page whose message happens to say "View Timeout Settings".

When several entries share a label (on / off / view / update variants of one setting), the
action's wording picks the variant. Returns None when nothing matches, so callers fall back
to BM25.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

EXACT_SCORE = 1.0  # a candidate screen name equals a catalog label
CONTAINED_SCORE = 0.9  # a multi-word catalog label appears verbatim in the action text

# Entries whose "label" is a URL type rather than a screen (DL-0294/0295, message "Onurl"/"Offurl").
_JUNK_KEYS = {"onurl", "offurl"}

_LEADING_VERBS = re.compile(
    r"^(?:turn on|turn off|switch on|switch off|enable|disable|activate|deactivate|open|adjust|"
    r"configure|change|check|set|view|manage|run|update|toggle|reset|clear|increase|decrease|"
    r"lower|raise|show|hide|use|access|review|go to)\s+(?:the\s+|your\s+|my\s+)?",
    re.IGNORECASE,
)
_STEP_OBJECT = re.compile(
    r"^(?:navigate to and open|navigate to|go to|open|tap on|tap|select|choose|toggle on|toggle off|toggle|"
    r"turn on|turn off|switch on|switch off|enable|disable)\s+(?:the\s+|your\s+|my\s+)?(.+)$",
    re.IGNORECASE,
)
_STEP_TAIL = re.compile(r"\s+(?:to|then|for|in|under|from|so|if|until|seven)\s+.*$|[.,;:!?].*$", re.IGNORECASE)

_ON = re.compile(r"\b(?:enable|enables|turn on|switch on|activate|allow)\b", re.IGNORECASE)
_OFF = re.compile(r"\b(?:disable|disables|turn off|switch off|deactivate|stop)\b", re.IGNORECASE)
_UPDATE = re.compile(r"\b(?:adjust|change|set|increase|decrease|lower|raise|update|drag)\b", re.IGNORECASE)
_VIEW = re.compile(r"\b(?:open|view|check|configure|manage|review|access)\b", re.IGNORECASE)

_TYPE_PREFERENCE = {
    "on": ["onURL", "onClickURL", "updateURL"],
    "off": ["offURL", "onClickURL", "updateURL"],
    "update": ["updateURL", "onClickURL", "onURL"],
    "view": ["onClickURL", "updateURL", "onURL"],
}

_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text.lower())  # "Back up data (TechCorp Cloud)" -> "back up data"
    text = re.sub(r"\bwi[\s-]*fi\b", "wifi", text)
    return " ".join(_WORD.findall(text))


def _intent(action_name: str, steps: List[str], description: str) -> str:
    """Action name first, then steps, then description: the most explicit wording wins."""
    for text in [action_name, " ".join(steps), description]:
        if _OFF.search(text):
            return "off"
        if _ON.search(text):
            return "on"
        if _UPDATE.search(text):
            return "update"
        if _VIEW.search(text):
            return "view"
    return "view"


def _destination_step(steps: List[str]) -> Optional[str]:
    """
    Object of the last navigation/toggle step: the screen the action ends on. Earlier steps
    are the path ("Tap Accessibility." on the way to Assistant menu) and must not match.
    """
    for step in reversed(steps):
        m = _STEP_OBJECT.match(step.strip())
        if m:
            return _STEP_TAIL.sub("", m.group(1))
    return None


def _candidate_phrases(action_name: str, steps: List[str]) -> List[str]:
    """Screen names in priority order: the action name, then the destination step."""
    phrases = [_norm(_LEADING_VERBS.sub("", action_name.strip(), count=1))]
    destination = _destination_step(steps)
    if destination:
        phrases.append(_norm(destination))
    return [p for p in phrases if p]


class _LabelIndex:
    def __init__(self, catalog: List[Dict[str, Any]]):
        self.by_label: Dict[str, List[Dict[str, Any]]] = {}
        for entry in catalog:
            if str(entry.get("deeplink", "")).endswith("://dummy_positive"):
                continue
            key = (entry.get("validation") or {}).get("key")
            label = _norm(key) if key else ""
            if not label or label in _JUNK_KEYS:
                continue
            self.by_label.setdefault(label, []).append(entry)


_INDEX_CACHE: Dict[int, _LabelIndex] = {}


def _index_for(catalog: List[Dict[str, Any]]) -> _LabelIndex:
    idx = _INDEX_CACHE.get(id(catalog))
    if idx is None:
        idx = _INDEX_CACHE[id(catalog)] = _LabelIndex(catalog)
    return idx


def is_junk_entry(entry: Dict[str, Any]) -> bool:
    key = (entry.get("validation") or {}).get("key")
    return bool(key) and _norm(key) in _JUNK_KEYS


def match_by_label(
    catalog: List[Dict[str, Any]],
    action_name: str,
    description: str = "",
    steps: Optional[List[str]] = None,
) -> Optional[Tuple[Dict[str, Any], float]]:
    """Returns (catalog entry, score) when the action names a catalog Settings label, else None."""
    steps = [s for s in (steps or []) if isinstance(s, str)]
    idx = _index_for(catalog)

    # 1. Exact: a candidate screen name equals a label. Earlier phrase = higher priority.
    label, score = None, 0.0
    for phrase in _candidate_phrases(action_name, steps):
        if phrase in idx.by_label:
            label, score = phrase, EXACT_SCORE
            break

    # 2. Contained: the longest multi-word label that appears verbatim in the action name,
    #    description or destination step (not in path steps, for the same reason as above).
    if label is None:
        text = f" {_norm(' '.join([action_name, description, _destination_step(steps) or '']))} "
        contained = [lb for lb in idx.by_label if " " in lb and f" {lb} " in text]
        if contained:
            label, score = max(contained, key=len), CONTAINED_SCORE

    if label is None:
        return None

    entries = idx.by_label[label]
    preference = _TYPE_PREFERENCE[_intent(action_name, steps, description)]
    candidates = [e for e in entries if e.get("originalType") in preference]
    if not candidates:
        return None  # e.g. only an "off" toggle exists for an "on" action: let BM25 decide

    # Preferred type first; among equals, the entry sharing the most words with the action.
    action_words = set(_WORD.findall(" ".join([action_name, description] + steps).lower()))

    def rank(e: Dict[str, Any]) -> Tuple[int, int]:
        entry_words = set(_WORD.findall(f"{e.get('description', '')} {e.get('qna_description', '')}".lower()))
        return (preference.index(e["originalType"]), -len(action_words & entry_words))

    return min(candidates, key=rank), score
