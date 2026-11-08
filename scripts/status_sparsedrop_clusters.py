#!/usr/bin/env python3
"""Read both assigned clusters once; save a provenance snapshot without checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

try:
    from sparsedrop_cluster_support import REMOTE_ROOT, array_ids, load_assignment
except ImportError:
    from scripts.sparsedrop_cluster_support import REMOTE_ROOT, array_ids, load_assignment


# This code runs only after the SSH command has entered REMOTE_ROOT. It reads
# small JSON files and the last 256 KiB of training logs, never checkpoint data.
REMOTE_READER = r'''
import datetime
import json
import os
from pathlib import Path
import subprocess

def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "stdout": "", "stderr": str(error)}

def read_json(path, errors):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, ValueError) as error:
        errors.append(f"{path}: {error}")
        return {}

rows = []
for assignment in PAYLOAD["assignments"]:
    directory = Path("results_fineweb20b") / assignment["name"]
    errors = []
    summary = read_json(directory / "summary.json", errors)
    config = read_json(directory / "config.json", errors)
    last_train = {}
    last_progress = {}
    log = directory / "train.jsonl"
    if log.exists():
        try:
            with log.open("rb") as source:
                source.seek(0, 2)
                size = source.tell()
                source.seek(max(0, size - 262144))
                lines = source.read().decode("utf-8", errors="replace").splitlines()
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if not last_progress and isinstance(event.get("step"), int):
                    last_progress = {key: event[key] for key in ("event", "time_utc", "step", "tokens_seen", "actual_tokens_seen") if key in event}
                if event.get("event") == "train":
                    last_train = {key: event[key] for key in ("time_utc", "step", "loss", "grad_norm_preclip", "tokens_seen", "tokens_per_second") if key in event}
                    break
        except OSError as error:
            errors.append(str(error))
    compact_summary = {key: summary[key] for key in ("status", "step", "actual_steps", "tokens_seen", "actual_tokens_seen", "effective_target_steps", "overshoot_steps") if key in summary}
    rows.append({"index": assignment["index"], "name": assignment["name"],
                 "summary": compact_summary, "last_train": last_train, "last_progress": last_progress,
                 "target_steps": config.get("effective_target_steps", config.get("arguments", {}).get("stop_after_steps")),
                 "sparsity_saved": (directory / "sparsity_final.json").is_file(),
                 "result_exists": directory.is_dir(), "errors": errors})

gates = []
for phase, job_id in PAYLOAD.get("gates", {}).items():
    errors = []
    path = Path("cluster") / f"stage_gate_{phase}.state.json"
    state = read_json(path, errors)
    if state and state.get("phase") != phase:
        errors.append(f"{path}: phase mismatch")
        state = {}
    gates.append({"phase": phase, "job_id": job_id, "state_file": str(path),
                  "state": state, "errors": errors})
arrays = PAYLOAD["arrays"]
known_jobs = arrays + [gate["job_id"] for gate in gates]
if known_jobs:
    # Completed arrays disappear from squeue; querying the user avoids treating
    # a completed array ID as a failed status check.
    queue = command(["squeue", "-r", "-h", "-u", str(os.getuid()), "-o", "%i|%T|%M|%R|%j"])
    queue["stdout"] = "\n".join(line for line in queue["stdout"].splitlines()
                                if line.split("|", 1)[0].split("_", 1)[0] in known_jobs)
    accounting = command(["sacct", "--array", "--starttime=1970-01-01", "-X", "-n", "-P", "-j", ",".join(known_jobs),
                          "--format=JobID,State,Elapsed,ExitCode,Start,End"])
else:
    queue = accounting = {"returncode": -1, "stdout": "", "stderr": "No registered Slurm arrays for this host"}
print(json.dumps({"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "queue": queue, "accounting": accounting, "runs": rows, "gates": gates}))
'''


def _job_records(output: str, *, accounting: bool) -> dict[int, list[dict]]:
    result = {}
    for line in output.splitlines():
        fields = line.strip().split("|")
        match = re.fullmatch(r"(\d+)_(\d+)", fields[0])
        if not match or len(fields) < (6 if accounting else 5):
            continue
        record = {"job_id": fields[0], "array_id": match[1], "state": fields[1].strip(),
                  "elapsed": fields[2]}
        if accounting:
            record.update(exit_code=fields[3], start=fields[4], end=fields[5])
        else:
            record.update(reason=fields[3], job_name=fields[4])
        result.setdefault(int(match[2]), []).append(record)
    return result


def _gate_record(output: str, job_id: str, *, accounting: bool) -> dict | None:
    for line in output.splitlines():
        fields = line.strip().split("|")
        if fields[0] != job_id or len(fields) < (6 if accounting else 5):
            continue
        record = {"job_id": job_id, "state": fields[1].strip(), "elapsed": fields[2]}
        if accounting:
            record.update(exit_code=fields[3], start=fields[4], end=fields[5])
        else:
            record.update(reason=fields[3], job_name=fields[4])
        return record
    return None


def combine_host(host: str, assignments: list[dict], remote: dict) -> dict:
    queue = _job_records(remote["queue"].get("stdout", ""), accounting=False)
    accounting = _job_records(remote["accounting"].get("stdout", ""), accounting=True)
    by_index = {row["index"]: row for row in remote["runs"]}
    output = []
    errors = []
    for key in ("queue", "accounting"):
        if remote[key]["returncode"]:
            errors.append(f"{key}: {remote[key].get('stderr', '')}")
    for assignment in assignments:
        row = dict(by_index.get(assignment["index"], {}))
        row.update(index=assignment["index"], name=assignment["name"], host=host, stage=assignment.get("stage"))
        index = assignment["index"]
        # If a replacement array exists for this index, newest array takes priority.
        live = sorted(queue.get(index, []), key=lambda item: int(item["array_id"]))
        accounted = sorted(accounting.get(index, []), key=lambda item: int(item["array_id"]))
        row["slurm"] = live[-1] if live else accounted[-1] if accounted else None
        summary = row.get("summary", {})
        summary_status = summary.get("status")
        if summary_status in ("complete", "stopped_over_budget"):
            status = summary_status
        elif live:
            status = live[-1]["state"].lower().split("+")[0]
        elif row["slurm"]:
            state = row["slurm"]["state"].lower().split("+")[0].split()[0]
            status = "finished_without_summary" if state == "completed" else state
        else:
            status = "unknown"
        row["status"] = status
        progress = row.get("last_progress", {})
        train = row.get("last_train", {})
        row["actual_steps"] = summary.get("actual_steps", summary.get("step", progress.get("step", train.get("step"))))
        tokens = summary.get("actual_tokens_seen", summary.get("tokens_seen",
                    progress.get("actual_tokens_seen", progress.get("tokens_seen", train.get("tokens_seen")))))
        row["training_tokens_B"] = tokens / 1e9 if isinstance(tokens, (int, float)) else None
        output.append(row)
    gates = []
    for source in remote.get("gates", []):
        gate = dict(source, host=host)
        gate["errors"] = list(source.get("errors", []))
        job_id = gate["job_id"]
        gate["slurm"] = (_gate_record(remote["queue"].get("stdout", ""), job_id, accounting=False)
                         or _gate_record(remote["accounting"].get("stdout", ""), job_id, accounting=True))
        slurm = gate["slurm"] or {}
        gate["status"] = slurm.get("state", "UNKNOWN").lower().split("+")[0].split()[0]
        gate["gate_status"] = gate.get("state", {}).get("status", "unknown")
        gate["release_confirmed"] = (gate["status"] == "completed" and slurm.get("exit_code") == "0:0"
                                      and gate["gate_status"] == "released" and not gate["errors"])
        if gate["status"] == "completed" and not gate["release_confirmed"]:
            gate["status"] = "completed_without_release_confirmation"
            gate["errors"].append("CPU gate finished without a verified successful release state")
        if gate["status"] in ("failed", "cancelled", "timeout", "node_fail", "out_of_memory", "unknown"):
            gate["errors"].append(f"CPU gate scheduler status: {gate['status']}")
        if gate["gate_status"] in ("blocked", "timed_out"):
            gate["errors"].append(f"CPU gate state: {gate['gate_status']}: {gate.get('state', {}).get('error', '')}")
        errors.extend(f"gate {gate['phase']} ({job_id}): {error}" for error in gate["errors"])
        gates.append(gate)
    return {"host": host, "reachable": True, "time_utc": remote["time_utc"],
            "errors": errors, "runs": output, "gates": gates,
            "gate_counts": dict(Counter(gate["status"] for gate in gates)),
            "queue": remote["queue"], "accounting": remote["accounting"],
            "counts": dict(Counter(row["status"] for row in output))}


def query_host(host: str, document: dict, timeout: int = 100) -> dict:
    assignments = [row for row in document["assignments"] if row["host"] == host]
    gates = {}
    try:
        for phase, job_id in document.get(f"stage_gate_jobs_{host}", {}).items():
            if phase not in ("p02_to_p005", "p005_to_schedule") or not re.fullmatch(r"\d+", str(job_id)):
                raise ValueError(f"Invalid CPU gate assignment: {phase!r}, {job_id!r}")
            gates[phase] = str(job_id)
        payload = {"assignments": assignments, "arrays": array_ids(document, host), "gates": gates}
        program = "PAYLOAD = " + repr(payload) + "\n" + REMOTE_READER
        process = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
             "-o", "ServerAliveCountMax=2", host, f"cd {shlex.quote(REMOTE_ROOT)} && venv/bin/python -"],
            input=program, text=True, capture_output=True, timeout=timeout)
        if process.returncode:
            raise RuntimeError(process.stderr.strip()[-4000:] or f"SSH exit {process.returncode}")
        remote = json.loads(process.stdout)
        return combine_host(host, assignments, remote)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        # Offline is unknown, never pending/finished inferred from a stale cache.
        rows = [{**row, "status": "unknown", "actual_steps": None, "training_tokens_B": None}
                for row in assignments]
        gate_rows = [{"host": host, "phase": phase, "job_id": job_id, "status": "unknown",
                      "gate_status": "unknown", "release_confirmed": False} for phase, job_id in gates.items()]
        return {"host": host, "reachable": False, "errors": [str(error)],
                "runs": rows, "counts": {"unknown": len(rows)},
                "gates": gate_rows, "gate_counts": {"unknown": len(gate_rows)} if gate_rows else {}}


def main(argv=None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, default=repository / "cluster/host_assignment.json")
    parser.add_argument("--manifest", type=Path, default=repository / "configs/sparsedrop_long/manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        document = load_assignment(args.assignment, args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Invalid cluster ownership: {error}\n")
    hosts = sorted({row["host"] for row in document["assignments"]})
    reports = {}
    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {executor.submit(query_host, host, document): host for host in hosts}
        for future in as_completed(futures):
            host = futures[future]
            reports[host] = future.result()
            print(f"{host}: " + json.dumps({"reachable": reports[host]["reachable"],
                                           "counts": reports[host]["counts"],
                                           "gate_counts": reports[host]["gate_counts"],
                                           "errors": reports[host]["errors"]}, ensure_ascii=False), flush=True)
    now = datetime.now(timezone.utc)
    snapshot = {"time_utc": now.isoformat(), "time_new_york": now.astimezone(ZoneInfo("America/New_York")).isoformat(),
                "assignment": document, "hosts": reports,
                "gate_counts": dict(Counter(gate["status"] for report in reports.values() for gate in report["gates"])),
                "counts": dict(Counter(row["status"] for report in reports.values() for row in report["runs"]))}
    output = args.output or repository.parent / "local_results/fineweb20b/provenance" / f"cluster_status_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    for report in reports.values():
        for gate in report["gates"]:
            print(json.dumps({key: gate.get(key) for key in ("host", "phase", "job_id", "status", "gate_status", "release_confirmed", "errors")}, ensure_ascii=False))
        for row in report["runs"]:
            if row["status"] in ("running", "completing", "suspended"):
                print(json.dumps({key: row.get(key) for key in ("host", "index", "name", "status", "actual_steps", "training_tokens_B", "last_train")}, ensure_ascii=False))
    print(json.dumps({"counts": snapshot["counts"], "snapshot": str(output)}, ensure_ascii=False))
    return int(any(not report["reachable"] or report["errors"] for report in reports.values()))


if __name__ == "__main__":
    raise SystemExit(main())
