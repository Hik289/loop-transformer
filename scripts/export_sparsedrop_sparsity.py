#!/usr/bin/env python3
"""Backfill validated final sparsity artifacts in direct result subdirectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.sparsedrop_export import export_final_sparsity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="Inspect direct run directories only; never recurse into archives")
    parser.add_argument("--strict", action="store_true", help="Return nonzero if an existing final summary cannot be validated/exported; runs without summaries are always skipped")
    args = parser.parse_args(argv)
    if not args.results_root.is_dir():
        parser.error(f"Results root is not a directory: {args.results_root}")
    counts = {"exported": 0, "skipped_without_summary": 0, "failed": 0}
    for run_dir in sorted(args.results_root.iterdir()):
        name = run_dir.name.lower()
        if (not run_dir.is_dir() or run_dir.is_symlink()
                or name in {"archive", "archives", "archived", "backup", "backups", "recovery"}
                or name.startswith(("archive_", "archive-", "before_100k_cap_", "."))):
            continue
        if not (run_dir / "summary.json").is_file():
            counts["skipped_without_summary"] += 1
            continue
        try:
            payload = export_final_sparsity(run_dir)
        except (OSError, ValueError) as error:
            counts["failed"] += 1
            print(json.dumps({"run": run_dir.name, "export": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        else:
            counts["exported"] += 1
            print(json.dumps({"run": run_dir.name, "export": "ok", "status": payload["status"],
                              "actual_steps": payload["actual_steps"], "training_tokens_B": payload["training_tokens_B"],
                              "rows": payload["coverage"]["observed_rows"], "budget_exceeded": payload["budget_exceeded"]}, ensure_ascii=False))
    print(json.dumps({"results_root": str(args.results_root.resolve()), **counts}, ensure_ascii=False))
    return 1 if args.strict and counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
