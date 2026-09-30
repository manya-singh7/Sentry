"""
Query dataset resolution shared by eval/run_eval.py and eval/ablation.py.

Resolution order when no path is given:
  1. queries.json in the repo root (a labelled set, if the team adds one)
  2. eval/official/input.txt (the official hackathon queries, one per line)
  3. queries.sample.json (6 hand-written sample queries; flagged as sample data)

An explicit --queries / --dataset path that does not exist is an error, never a silent
fall back to sample data.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_INPUT = ROOT / "eval" / "official" / "input.txt"


def find_queries_path(explicit_path: str = "") -> Tuple[Path, bool]:
    """Returns (path, is_sample)."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Query dataset not found: {explicit_path}")
        return p, "sample" in p.name.lower()

    for path, is_sample in [
        (ROOT / "queries.json", False),
        (OFFICIAL_INPUT, False),
        (ROOT / "queries.sample.json", True),
    ]:
        if path.exists():
            return path, is_sample
    raise FileNotFoundError("No query dataset found (queries.json, eval/official/input.txt, queries.sample.json).")


def results_meta_path(results_path: Path) -> Path:
    """results.jsonl -> results.meta.json"""
    return results_path.with_name(results_path.stem + ".meta.json")


def write_results_meta(results_path: Path, queries_path: Path, queries_are_sample: bool,
                       n_queries: int, live: bool) -> None:
    """Sidecar describing the datasets behind a results.jsonl (results.jsonl itself stays Appendix B)."""
    import sys
    from datetime import datetime, timezone

    sys.path.insert(0, str(ROOT / "backend"))
    from retrieval import get_retriever

    retriever = get_retriever()

    def rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    meta = {
        "queries_file": rel(queries_path),
        "queries_are_sample": queries_are_sample,
        "n_queries": n_queries,
        "catalog_file": rel(retriever.catalog_path),
        "catalog_is_sample": retriever.is_sample,
        "live": live,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    results_meta_path(Path(results_path)).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def read_results_meta(results_path: Path) -> Dict[str, Any]:
    p = results_meta_path(Path(results_path))
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """
    .txt: one query per non-empty line -> [{"id": "q1", "query": ...}] (no domain labels).
    .json: a list of query objects, or {"queries": [...]}.
    """
    if queries_path.suffix.lower() == ".txt":
        lines = [line.strip() for line in queries_path.read_text(encoding="utf-8").splitlines()]
        return [{"id": f"q{i}", "query": line} for i, line in enumerate((l for l in lines if l), start=1)]

    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    return []
