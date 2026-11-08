#!/usr/bin/env python3
"""Fill original SparseDrop README tables from each validated final experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.sparsedrop_readme import update_readme_results


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repository / "configs/sparsedrop_long/manifest.json")
    parser.add_argument("--results-root", type=Path, required=True, help="Read only direct experiment directories here; never use configured remote output_dir")
    parser.add_argument("--readme", type=Path, default=repository.parent / "README_SparseDrop_Loop_Ouro_Metrics_Formulas_CN.md")
    parser.add_argument("--live-readme", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit 1 if a final result is invalid or declares another target budget")
    args = parser.parse_args(argv)
    try:
        result = update_readme_results(manifest_path=args.manifest, results_root=args.results_root,
                                      readme_path=args.readme, live_readme_path=args.live_readme,
                                      status_path=args.status_json)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"README update failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"counts": result["counts"], **{key: result[key] for key in ("readme_path", "live_readme_path", "status_path")}}, ensure_ascii=False))
    for row in result["rows"]:
        for error in row["errors"]:
            print(f"{row['name']}: {error}", file=sys.stderr)
    return int(args.strict and bool(result["counts"]["invalid"] or result["counts"]["budget_mismatch"]))


if __name__ == "__main__":
    raise SystemExit(main())
