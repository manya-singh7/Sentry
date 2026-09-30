"""
Deeplink retrieval accuracy on the official deeplinks.json (offline, no Gemini).

Each case is an action the pipeline would plausibly produce for the input.txt complaints,
labelled with the catalog entries that are correct for it (or "dummy" when the catalog has no
entry for that screen). Labels are hand-picked from the catalog descriptions; review and extend
them as needed.

Runs the real main.get_deeplinks() and reports, per case and overall:
  correct       served link is one of the labelled entries
  wrong         a catalog link was served, but not a labelled one (the user gets the wrong screen)
    polarity    ...and it is the on/off sibling of the labelled entry (e.g. Disable instead of Enable)
  missed        a labelled catalog entry exists, but the dummy / no link was served
  dummy ok/bad  for screens with no catalog entry: dummy served, or a wrong catalog link served

Usage: python eval/official/eval_retrieval.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from main import get_deeplinks  # noqa: E402
from schema import DUMMY_DEEPLINK_URI, ActionCategory  # noqa: E402

CATALOG = json.loads((ROOT / "deeplinks.json").read_text())["deeplinks"]
BY_ID = {e["id"]: e for e in CATALOG}
BY_URI = {e["deeplink"]: e for e in CATALOG}

S = ["Open Settings."]
# (action name, description, steps, labelled catalog IDs or "dummy")
CASES = [
    ("Adjust Screen Brightness", "It will change display brightness level", S + ["Tap Display.", "Drag the Brightness slider."], {"DL-0232", "DL-0496"}),
    ("Disable Adaptive Brightness", "It will stop automatic brightness changes", S + ["Tap Display.", "Turn off Adaptive brightness."], {"DL-0020"}),
    ("Enable Adaptive Brightness", "It will adjust brightness to surrounding light", S + ["Tap Display.", "Turn on Adaptive brightness."], {"DL-0021"}),
    ("Adjust Screen Timeout", "It will keep the screen on longer", S + ["Tap Display.", "Tap Screen timeout."], {"DL-0220"}),
    ("Adjust Refresh Rate", "It will change motion smoothness setting", S + ["Tap Display.", "Tap Motion smoothness."], {"DL-0228"}),
    ("Disable Always On Display", "It will turn off always on display", S + ["Tap Lock screen.", "Turn off Always On Display."], {"DL-0482"}),
    ("Enable Touch Sensitivity", "It will improve touch response on screen", S + ["Tap Display.", "Turn on Touch sensitivity."], {"DL-0126"}),
    ("Adjust Screen Zoom", "It will make screen content larger", S + ["Tap Display.", "Tap Screen zoom."], {"DL-0229", "DL-0549"}),
    ("Change Screen Mode", "It will change the display color profile", S + ["Tap Display.", "Tap Screen mode."], {"DL-0215"}),
    ("Disable Extra Dim", "It will restore normal screen brightness", S + ["Tap Accessibility.", "Turn off Extra dim."], {"DL-0203", "DL-0190"}),  # both are Extra dim
    ("Disable Eye Comfort Shield", "It will turn off blue light filter", S + ["Tap Display.", "Turn off Eye comfort shield."], {"DL-0039"}),
    ("Back Up Phone Data", "It will facilitate secure data transfer between", S + ["Tap on Accounts and backup.", "Select Back up data."], {"DL-0542"}),
    ("Optimize Device Performance", "It will clean memory and close apps", S + ["Tap Device care.", "Tap Optimize now."], {"DL-0538"}),
    ("Run Device Diagnostics", "It will check device hardware status", ["Open TechCorp Members.", "Tap Diagnostics."], {"DL-0478"}),
    ("Check Warranty Coverage", "It will show warranty period and coverage", ["Open TechCorp Warranty Care.", "Tap Warranty details."], {"DL-0577"}),
    ("Update System Apps", "It will install latest system app updates", S + ["Tap Software update."], {"DL-0115"}),
    ("Disable Color Inversion", "It will restore normal screen colors", S + ["Tap Accessibility.", "Turn off Color inversion."], {"DL-0280"}),
    ("Configure Navigation Bar", "It will let you choose navigation type", S + ["Tap Display.", "Tap Navigation bar."], {"DL-0169"}),
    ("Disable Screen Flash Notification", "It will stop the screen flashing for alerts", S + ["Tap Accessibility.", "Turn off Screen flash notification."], {"DL-0210"}),
    ("Enable Dim Strobing", "It will reduce flashing light effects", S + ["Tap Accessibility.", "Turn on Dim strobing."], {"DL-0206"}),
    ("Enable Power Saving Mode", "It will extend battery life", S + ["Tap Battery.", "Turn on Power saving."], {"DL-0412"}),
    ("Check Battery Health", "It will diagnose battery status", ["Open TechCorp Members.", "Tap Battery status."], {"DL-0477"}),
    ("Turn On Bluetooth", "It will enable Bluetooth connection", S + ["Tap Connections.", "Turn on Bluetooth."], {"DL-0495"}),
    ("Turn On Wi-Fi", "It will enable wireless network connection", S + ["Tap Connections.", "Turn on Wi-Fi."], {"DL-0574"}),
    ("Enable Mobile Data", "It will turn on cellular data", S + ["Tap Connections.", "Turn on Mobile data."], {"DL-0082"}),
    ("Turn Off Airplane Mode", "It will restore wireless connections", S + ["Tap Connections.", "Turn off Airplane mode."], {"DL-0274"}),
    ("Enable Keep Screen On While Viewing", "It will keep screen on while viewing", S + ["Tap Advanced features.", "Turn on Keep screen on while viewing."], {"DL-0259"}),
    ("Disable Auto Dim Screen", "It will stop screen dimming on low battery", S + ["Tap Battery.", "Turn off Auto dim screen."], {"DL-0401"}),
    ("Open One-handed Mode", "It will turn off the shrunken screen", S + ["Tap Advanced features.", "Tap One-handed mode."], {"DL-0537"}),
    # Screens with no catalog entry: the dummy is the correct answer.
    ("Configure Game Booster", "It will open game booster settings", S + ["Tap Advanced features.", "Tap Game Booster."], "dummy"),
    ("Disable Assistant Menu", "It will remove the floating shortcut circle", S + ["Tap Accessibility.", "Tap Interaction and dexterity.", "Turn off Assistant menu."], "dummy"),
    ("Configure Edge Lighting", "It will change edge lighting style", S + ["Tap Notifications.", "Tap Edge lighting style."], "dummy"),
    ("Adjust Cover Screen Settings", "It will configure the cover screen display", S + ["Tap Cover screen."], "dummy"),
]


def _sibling(a: dict, b: dict) -> bool:
    """On/off pair: same validation key, opposite originalType."""
    ka = (a.get("validation") or {}).get("key")
    kb = (b.get("validation") or {}).get("key")
    return bool(ka) and ka == kb and {a.get("originalType"), b.get("originalType")} == {"onURL", "offURL"}


def main() -> None:
    counts = {"correct": 0, "wrong": 0, "polarity": 0, "missed": 0, "dummy_ok": 0, "dummy_bad": 0}
    rows = []
    for name, desc, steps, expected in CASES:
        dl, score = get_deeplinks(name, desc, category=ActionCategory.auto, steps=steps, return_score=True)
        uri = dl.deeplink if dl else None
        got = BY_URI.get(uri, {}) if uri and uri != DUMMY_DEEPLINK_URI else None
        got_label = got["id"] + " " + got["message"] if got else ("dummy" if uri == DUMMY_DEEPLINK_URI else "no link")

        if expected == "dummy":
            verdict = "dummy_ok" if uri == DUMMY_DEEPLINK_URI else "dummy_bad"
        elif got and got["id"] in expected:
            verdict = "correct"
        elif got:
            verdict = "wrong"
            if any(_sibling(got, BY_ID[e]) for e in expected):
                counts["polarity"] += 1
                verdict = "wrong (polarity)"
        else:
            verdict = "missed"
        counts[verdict.split(" ")[0]] += 1
        want = "dummy" if expected == "dummy" else ", ".join(sorted(expected))
        rows.append((verdict, score, name, got_label, want))

    print(f"{'verdict':17} {'score':>6}  {'action':36} {'served':44} expected")
    print("-" * 130)
    for verdict, score, name, got_label, want in rows:
        mark = "OK " if verdict in ("correct", "dummy_ok") else "XX "
        print(f"{mark}{verdict:14} {score if score is not None else 0:6.3f}  {name:36} {got_label[:44]:44} {want}")

    n_cat = sum(1 for c in CASES if c[3] != "dummy")
    n_dummy = len(CASES) - n_cat
    print(f"\nCatalog screens ({n_cat}): {counts['correct']} correct, {counts['wrong']} wrong link served "
          f"({counts['polarity']} of them on/off polarity), {counts['missed']} missed")
    print(f"Unindexed screens ({n_dummy}): {counts['dummy_ok']} dummy correctly, {counts['dummy_bad']} wrong catalog link served")
    total_ok = counts["correct"] + counts["dummy_ok"]
    print(f"Overall: {total_ok}/{len(CASES)} = {total_ok / len(CASES):.0%} correct; "
          f"{counts['wrong'] + counts['dummy_bad']} cases would send the user to the wrong Settings screen")


if __name__ == "__main__":
    main()
