#!/usr/bin/env python3
"""Release the next SparseDrop stage only after both clusters finish successfully.

This CPU-only gate relies on the unchanged training job's exit contract: final
summary, exact token budget and final sparsity are validated before exit zero.
It never interprets a missing queue entry as completion and never submits jobs.

Plan schema (all 12 explicit array elements are required in each list):
{
  "schema_version": 1,
  "phase": "p02_to_p005",
  "owner": "researcher",
  "requirements": [
    {"cluster": "h200cluster", "job_ids": ["53342_11", "..."]},
    {"cluster": "h100cluster", "job_ids": ["NEW_ARRAY_12", "..."]}
  ],
  "release_targets": [
    {"cluster": "h200cluster", "job_ids": ["53342_5", "..."]},
    {"cluster": "h100cluster", "job_ids": ["NEW_ARRAY_6", "..."]}
  ]
}
The second phase is p005_to_schedule. Each cluster owns six tasks per stage.
The state and exclusive lock default to <plan>.state.json and <state>.lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


CLUSTERS = {"h100cluster", "h200cluster"}
STAGES = {
    "p02": {11, 12, 13, 21, 22, 23, 27, 28, 29, 49, 50, 51},
    "p005": {5, 6, 7, 15, 16, 17, 24, 25, 26, 31, 32, 33},
    "schedule": {35, 36, 37, 38, 40, 41, 42, 43, 45, 46, 47, 48},
}
PHASES = {"p02_to_p005": ("p02", "p005"),
          "p005_to_schedule": ("p005", "schedule")}
FAILED = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
          "OUT_OF_MEMORY", "OOM", "PREEMPTED", "REVOKED", "SPECIAL_EXIT", "TIMEOUT"}
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
          "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING",
          "STAGE_OUT"}
JOB_ID = re.compile(r"([1-9][0-9]*)_([0-9]+)\Z")


class GateError(RuntimeError):
    """Permanent validation error: do not release anything further."""


class RetryableError(RuntimeError):
    """Transport/controller failure: retry without opening the barrier."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temp.exists():
            temp.unlink()


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or plan.get("phase") not in PHASES:
        raise GateError("Expected schema_version=1 and a supported phase")
    if plan.get("owner") != "researcher":
        raise GateError("This project gate is restricted to owner researcher")
    previous, following = PHASES[plan["phase"]]
    for key, stage in (("requirements", previous), ("release_targets", following)):
        groups = plan.get(key)
        if not isinstance(groups, list) or len(groups) != 2:
            raise GateError(f"{key} must contain exactly the two cluster groups")
        clusters, tasks = set(), []
        for group in groups:
            if not isinstance(group, dict) or group.get("cluster") not in CLUSTERS:
                raise GateError(f"{key} has an unexpected cluster")
            cluster = group["cluster"]
            if cluster in clusters:
                raise GateError(f"{key} repeats cluster {cluster}")
            clusters.add(cluster)
            ids = group.get("job_ids")
            if not isinstance(ids, list) or len(ids) != 6 or len(set(ids)) != 6:
                raise GateError(f"{key}/{cluster} must contain six unique explicit elements")
            for job_id in ids:
                match = JOB_ID.fullmatch(job_id) if isinstance(job_id, str) else None
                if not match:
                    raise GateError(f"Expected an explicit numeric array element: {job_id!r}")
                tasks.append(int(match[2]))
        if len(set(tasks)) != 12 or set(tasks) != STAGES[stage]:
            raise GateError(f"{key} must assign each {stage} task exactly once")
    return plan


def command(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=90,
                                check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RetryableError(f"Slurm command unavailable: {exc}") from exc
    if result.returncode:
        message = (result.stderr or result.stdout).strip()[:1500]
        raise RetryableError(f"{argv[0]} exit {result.returncode}: {message}")
    return result.stdout


def accounting_rows(output: str, cluster: str, expected: set[str],
                    owner: str) -> dict[str, dict[str, str]]:
    rows = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        values = line.strip().split("|")
        # --parsable2 normally omits the terminal separator; tolerate it.
        if len(values) == 6 and values[-1] == "":
            values.pop()
        if len(values) != 5:
            raise GateError(f"Malformed accounting record: {line!r}")
        got_cluster, job_id, got_owner, raw_state, exit_code = values
        if got_cluster != cluster or got_owner != owner or job_id not in expected:
            raise GateError(f"Accounting identity mismatch: {line!r}")
        state = raw_state.split()[0] if raw_state else ""
        row = {"cluster": cluster, "job_id": job_id, "owner": got_owner,
               "state": state, "exit_code": exit_code}
        if job_id in rows and rows[job_id] != row:
            raise GateError(f"Conflicting accounting records for {cluster}/{job_id}")
        rows[job_id] = row
    return rows


def array_task_ids(value: str) -> set[int]:
    """Decode Slurm's compressed ArrayTaskId without accepting unrelated tasks.

    Even `scontrol show job ARRAY_5` can return a shared pending record such as
    ArrayTaskId=5,7,15-17%2 until Slurm materializes that element. Commands still
    target the single explicit ARRAY_5; the shared record is only inspected.
    """
    match = re.fullmatch(r"([0-9,:-]+)(?:%[1-9][0-9]*)?", value)
    if match is None:
        raise GateError(f"Malformed controller ArrayTaskId: {value!r}")
    task_ids = set()
    for part in match[1].split(","):
        item = re.fullmatch(r"([0-9]+)(?:-([0-9]+)(?::([1-9][0-9]*))?)?", part)
        if item is None:
            raise GateError(f"Malformed controller array range: {part!r}")
        first = int(item[1])
        last = int(item[2]) if item[2] is not None else first
        stride = int(item[3]) if item[3] is not None else 1
        if not 0 <= first <= last <= 51:
            raise GateError(f"Controller array range outside this manifest: {part!r}")
        task_ids.update(range(first, last + 1, stride))
    return task_ids


def control_record(output: str, cluster: str, job_id: str,
                   owner: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise GateError(f"Expected exactly one controller record for {cluster}/{job_id}")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=(\S*)", lines[0]))
    array, task = job_id.split("_")
    if (fields.get("ArrayJobId") != array
            or int(task) not in array_task_ids(fields.get("ArrayTaskId", ""))):
        raise GateError(f"Controller returned wrong array element for {cluster}/{job_id}")
    if not re.fullmatch(re.escape(owner) + r"\([0-9]+\)", fields.get("UserId", "")):
        raise GateError(f"Controller returned wrong owner for {cluster}/{job_id}")
    if not fields.get("JobId", "").isdigit():
        raise GateError(f"Controller JobId is missing for {cluster}/{job_id}")
    if fields.get("Cluster", cluster) != cluster:
        raise GateError(f"Controller returned wrong cluster for {cluster}/{job_id}")
    if not fields.get("JobState") or "Reason" not in fields:
        raise GateError(f"Controller state/reason missing for {cluster}/{job_id}")
    return fields


class StageGate:
    def __init__(self, plan: dict[str, Any], state_path: Path, runner=command):
        self.plan = validate_plan(plan)
        self.state_path = state_path
        self.runner = runner
        self._local_h100_verified = False
        self.plan_hash = hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if state_path.exists():
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
            if self.state.get("plan_sha256") != self.plan_hash:
                raise GateError("State belongs to a different plan; refusing to reuse it")
        else:
            self.state = {"schema_version": 1, "phase": plan["phase"],
                          "plan_sha256": self.plan_hash, "created_at": now(),
                          "status": "waiting", "releases": {}}
        self.save()

    def save(self) -> None:
        self.state["updated_at"] = now()
        atomic_json(self.state_path, self.state)

    def control(self, cluster: str, *arguments: str) -> str:
        # The accounting DB advertises an H100 controller address that is not
        # reachable from this host. Its installed local Slurm configuration
        # works. Only use that default after proving which cluster it names.
        if cluster == "h100cluster":
            if not self._local_h100_verified:
                output = self.runner(["scontrol", "show", "config"])
                names = re.findall(r"^\s*ClusterName\s*=\s*(\S+)\s*$", output, flags=re.MULTILINE)
                if names != ["h100cluster"]:
                    raise GateError(f"Default Slurm controller is not verified h100cluster: {names!r}")
                self._local_h100_verified = True
            return self.runner(["scontrol", *arguments])
        if cluster != "h200cluster":
            raise GateError(f"Unsupported controller cluster: {cluster}")
        return self.runner(["scontrol", "-M", cluster, *arguments])

    def requirements_complete(self) -> bool:
        all_complete = True
        snapshot = []
        for group in self.plan["requirements"]:
            cluster, ids = group["cluster"], group["job_ids"]
            output = self.runner([
                "sacct", "-M", cluster, "--array", "-X", "-nP", "--starttime=1970-01-01",
                "-j", ",".join(ids), "--format=Cluster%32,JobID%64,User%64,State%32,ExitCode",
            ])
            rows = accounting_rows(output, cluster, set(ids), self.plan["owner"])
            for job_id in ids:
                row = rows.get(job_id)
                if row is None:
                    all_complete = False
                    snapshot.append({"cluster": cluster, "job_id": job_id, "state": "UNKNOWN"})
                    continue
                snapshot.append(row)
                state = row["state"]
                if state in FAILED or (state == "COMPLETED" and row["exit_code"] != "0:0"):
                    raise GateError(f"Prerequisite failed: {cluster}/{job_id}: {row}")
                if state != "COMPLETED":
                    all_complete = False
                    if state not in ACTIVE:
                        raise GateError(f"Unrecognized prerequisite state: {cluster}/{job_id}: {state}")
        self.state["prerequisites"] = snapshot
        self.state["last_checked_at"] = now()
        if all_complete:
            self.state["prerequisites_verified_at"] = now()
        self.save()
        return all_complete

    def inspect_target(self, cluster: str, job_id: str) -> tuple[str, dict[str, str]]:
        output = self.control(cluster, "--oneliner", "show", "job", job_id)
        record = control_record(output, cluster, job_id, self.plan["owner"])
        state, reason = record["JobState"], record["Reason"]
        key = f"{cluster}/{job_id}"
        if state in FAILED:
            raise GateError(f"Release target already failed: {key}: {state}")
        if state == "PENDING" and reason == "JobHeldUser":
            if record.get("Dependency") not in {"(null)", "None", ""}:
                raise GateError(f"Release target has unexpected dependency: {key}")
            return "held", record
        if state in {"RUNNING", "CONFIGURING", "COMPLETING", "COMPLETED"}:
            if state == "COMPLETED" and record.get("ExitCode") != "0:0":
                raise GateError(f"Release target completed with nonzero exit: {key}")
            return "already_released", record
        intent = self.state["releases"].get(key, {})
        if state == "PENDING" and intent.get("intent_at"):
            if ("Hold" in reason or "Held" in reason or reason in {"Dependency", "DependencyNeverSatisfied"}
                    or record.get("Dependency") not in {"(null)", "None", ""}):
                raise GateError(f"Previously released target has a new hold/dependency: {key}: {reason}")
            # Resource/account/QOS pending reasons vary between Slurm versions.
            # A durable intent plus a successful identity check is sufficient:
            # no release command is issued for this already-unheld task.
            return "already_released", record
        raise GateError(f"Target is not user-held or a known previous release: {key}: {state}/{reason}")

    def tick(self) -> bool:
        if self.state.get("status") == "released":
            return True
        if not self.requirements_complete():
            self.state["status"] = "waiting"
            self.save()
            return False
        targets = [(group["cluster"], job_id) for group in self.plan["release_targets"]
                   for job_id in group["job_ids"]]
        # Validate every target before changing the first target.
        for cluster, job_id in targets:
            self.inspect_target(cluster, job_id)
        self.state["status"] = "releasing"
        self.save()
        for cluster, job_id in targets:
            key = f"{cluster}/{job_id}"
            disposition, record = self.inspect_target(cluster, job_id)
            if disposition == "held":
                intent = self.state["releases"].setdefault(key, {})
                intent.setdefault("intent_at", now())
                intent["before"] = {"state": record["JobState"], "reason": record["Reason"]}
                self.save()  # Crash-safe write-ahead intent precedes the mutation.
                self.control(cluster, "release", job_id)
                disposition, record = self.inspect_target(cluster, job_id)
                if disposition == "held":
                    raise RetryableError(f"Release has not become visible: {key}")
            entry = self.state["releases"].setdefault(key, {})
            entry["verified_released_at"] = now()
            entry["observed_state"] = record["JobState"]
            self.save()
        self.state["status"] = "released"
        self.state["released_at"] = now()
        self.save()
        return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-seconds", type=int, default=603900)
    parser.add_argument("--once", action="store_true",
                        help="Check once; return 3 if waiting or transport unavailable")
    args = parser.parse_args(argv)
    if args.poll_seconds < 1 or args.max_seconds < 1:
        parser.error("poll-seconds and max-seconds must be positive")
    state_path = args.state_file or args.plan.with_suffix(".state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(state_path.name + ".lock")
    try:
        with lock_path.open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GateError("Another controller holds this plan's state lock") from exc
            gate = StageGate(json.loads(args.plan.read_text(encoding="utf-8")), state_path)
            started = time.monotonic()
            while time.monotonic() - started < args.max_seconds:
                try:
                    if gate.tick():
                        print(json.dumps({"phase": gate.plan["phase"], "status": "released",
                                          "state_file": str(state_path)}, ensure_ascii=False), flush=True)
                        return 0
                except RetryableError as exc:
                    gate.state["last_transport_error"] = {"at": now(), "error": str(exc)}
                    gate.save()
                    print(f"{now()} retrying: {exc}", file=sys.stderr, flush=True)
                except GateError as exc:
                    gate.state["status"] = "blocked"
                    gate.state["error"] = {"at": now(), "error": str(exc)}
                    gate.save()
                    raise
                if args.once:
                    return 3
                time.sleep(min(args.poll_seconds, max(0, args.max_seconds - (time.monotonic() - started))))
            gate.state["status"] = "timed_out"
            gate.save()
            raise GateError("Gate timeout; unreleased jobs remain held")
    except (GateError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STAGE GATE BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
