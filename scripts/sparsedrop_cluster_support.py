"""Small, standard-library ownership helpers for the two SparseDrop clusters."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

HOSTS = ("h200", "h100")
REMOTE_ROOT = os.environ.get("SPARSEDROP_REMOTE_ROOT", "/path/to/loop-transformer")
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def load_assignment(assignment_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    expected = {row["index"]: row["name"] for row in manifest}
    if len(expected) != len(manifest) or len(set(expected.values())) != len(manifest):
        raise ValueError("Manifest has duplicate indices or names")
    if assignment_path.exists():
        document = json.loads(assignment_path.read_text())
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported host assignment schema")
    else:
        document = {"schema_version": 1, "array_h200": 53342, "legacy_single_host": True,
                    "assignments": [dict(row, host="h200", stage="legacy") for row in manifest]}
    rows = document.get("assignments", [])
    indices, names = set(), set()
    for row in rows:
        index, name, host = row.get("index"), row.get("name"), row.get("host")
        if isinstance(index, bool) or not isinstance(index, int) or index not in expected:
            raise ValueError(f"Unexpected assignment index: {index!r}")
        if name != expected[index] or not SAFE_NAME.fullmatch(name) or name in (".", ".."):
            raise ValueError(f"Assignment name does not match manifest: {name!r}")
        if host not in HOSTS:
            raise ValueError(f"Unsupported host: {host!r}")
        if index in indices or name in names:
            raise ValueError(f"Duplicate assignment: {index}, {name}")
        indices.add(index)
        names.add(name)
    if indices != set(expected):
        raise ValueError("Host assignment must cover every manifest condition exactly once")
    return document


def claim_results(document: dict, results_root: Path) -> None:
    """Refuse cross-host overwrites, including adoption of old unmarked H200 data."""
    planned = []
    for row in document["assignments"]:
        directory = results_root / row["name"]
        if directory.is_symlink():
            raise ValueError(f"Refusing symlink result directory: {directory}")
        marker = directory / ".source_host.json"
        if marker.exists():
            current = json.loads(marker.read_text())
            if current.get("host") != row["host"] or current.get("index") != row["index"]:
                raise ValueError(f"Result ownership conflict: {directory}")
        elif row["host"] != "h200":
            # Before multi-host collection, every locally downloaded result was H200.
            recorded = any((directory / name).exists() and (directory / name).stat().st_size
                           for name in ("train.jsonl", "eval.jsonl", "summary.json", "last.pt", "best.pt"))
            if recorded:
                raise ValueError(f"Existing unmarked H200 result cannot be overwritten by {row['host']}: {directory}")
        planned.append((directory, marker, row))
    # Validate the entire mapping before writing any marker.
    for directory, marker, row in planned:
        directory.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"host": row["host"], "index": row["index"],
                                     "name": row["name"]}, ensure_ascii=False) + "\n")


def array_ids(document: dict, host: str) -> list[str]:
    """Accept one array, a list of arrays, or a stage->array mapping."""
    found = set()

    def add(value):
        if isinstance(value, dict):
            for child in value.values():
                add(child)
        elif isinstance(value, list):
            for child in value:
                add(child)
        elif value is not None and not isinstance(value, bool):
            token = str(value)
            if re.fullmatch(r"\d+(?:_\d+)?", token):
                found.add(token.split("_")[0])

    add(document.get(f"array_{host}"))
    add(document.get(f"arrays_{host}"))
    add(document.get(f"jobs_{host}_by_stage"))
    for row in document["assignments"]:
        if row["host"] == host:
            add(row.get("array_id"))
            add(row.get("job_id"))
    return sorted(found, key=int)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--claim-results", type=Path)
    args = parser.parse_args(argv)
    try:
        document = load_assignment(args.assignment, args.manifest)
        if args.claim_results:
            claim_results(document, args.claim_results)
        for row in sorted(document["assignments"], key=lambda row: row["index"]):
            print(f"{row['host']}\t{row['name']}\t{row['index']}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Invalid cluster ownership: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
