#!/usr/bin/env bash
# Set an explicit remote project directory before downloading records.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE="${SPARSEDROP_REMOTE_ROOT:?Set the remote project directory}"
python "$ROOT/scripts/pull_records.py" --study 400m --remote-root "$REMOTE" --host h100 --host h200 "$@"
