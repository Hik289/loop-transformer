"""Host ownership and read-only status checks; standard-library unittest only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.sparsedrop_cluster_support import array_ids, claim_results, load_assignment
from scripts.status_sparsedrop_clusters import REMOTE_READER, combine_host, query_host


class ClusterToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.assignment = self.root / "host_assignment.json"
        self.rows = [{"index": 0, "name": "run0"}, {"index": 3, "name": "run3"}]
        self.manifest.write_text(json.dumps(self.rows))
        self.document = {"schema_version": 1, "array_h200": 53342, "arrays_h100": [100, 101],
                         "assignments": [{**self.rows[0], "host": "h200", "stage": 1},
                                         {**self.rows[1], "host": "h100", "stage": 2}]}
        self.assignment.write_text(json.dumps(self.document))

    def tearDown(self):
        self.temp.cleanup()

    def test_assignment_requires_exact_manifest_coverage(self):
        for mutation in (
            lambda doc: doc["assignments"].pop(),
            lambda doc: doc["assignments"].append(doc["assignments"][0]),
            lambda doc: doc["assignments"][1].update(name="../escape"),
            lambda doc: doc["assignments"][1].update(host="foreign"),
        ):
            document = deepcopy(self.document)
            mutation(document)
            self.assignment.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_assignment(self.assignment, self.manifest)

    def test_legacy_single_h200_without_assignment(self):
        self.assignment.unlink()
        document = load_assignment(self.assignment, self.manifest)
        self.assertTrue(document["legacy_single_host"])
        self.assertEqual({row["host"] for row in document["assignments"]}, {"h200"})

    def test_ownership_conflict_cannot_overwrite(self):
        results = self.root / "results"
        claim_results(self.document, results)
        changed = deepcopy(self.document)
        changed["assignments"][0]["host"] = "h100"
        with self.assertRaisesRegex(ValueError, "ownership conflict"):
            claim_results(changed, results)
        self.assertEqual(json.loads((results / "run0/.source_host.json").read_text())["host"], "h200")

    def test_old_unmarked_h200_data_cannot_be_adopted_by_h100(self):
        directory = self.root / "results/run3"
        directory.mkdir(parents=True)
        (directory / "train.jsonl").write_text('{"step": 100}\n')
        with self.assertRaisesRegex(ValueError, "unmarked H200"):
            claim_results(self.document, directory.parent)
        self.assertFalse((directory.parent / "run0/.source_host.json").exists())

    def test_old_h200_data_and_new_h100_data_are_allowed(self):
        results = self.root / "results"
        (results / "run0").mkdir(parents=True)
        (results / "run0/summary.json").write_text("{}")
        claim_results(self.document, results)
        claim_results(self.document, results)
        self.assertEqual(json.loads((results / "run3/.source_host.json").read_text())["host"], "h100")

    def test_symlink_directory_refused(self):
        results = self.root / "results"
        results.mkdir()
        (results / "run0").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlink"):
            claim_results(self.document, results)

    def test_array_lists_stage_mapping_and_specific_jobs(self):
        self.document["jobs_h100_by_stage"] = {"2": 100, "3": 101, "4": 102}
        self.document["assignments"][1]["job_id"] = "103_3"
        self.assertEqual(array_ids(self.document, "h100"), ["100", "101", "102", "103"])

    def test_offline_host_is_unknown_not_pending(self):
        with patch("scripts.status_sparsedrop_clusters.subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 1)):
            result = query_host("h100", self.document)
        self.assertFalse(result["reachable"])
        self.assertEqual(result["counts"], {"unknown": 1})
        self.assertIsNone(result["runs"][0]["actual_steps"])

    def test_remote_command_enters_scratch_and_never_loads_checkpoint(self):
        with patch("scripts.status_sparsedrop_clusters.subprocess.run", side_effect=OSError("offline")) as run:
            query_host("h100", self.document)
        command = run.call_args.args[0]
        self.assertEqual(command[-1], "cd /path/to/loop-transformer && venv/bin/python -")
        compile(run.call_args.kwargs["input"], "<remote>", "exec")
        self.assertNotIn("torch.load", REMOTE_READER)
        self.assertNotIn("last.pt", REMOTE_READER)

    def test_accounting_expands_pending_array_elements(self):
        calls = []

        def fake_command(args, **kwargs):
            calls.append(args)
            output = "100_3|PENDING|00:00|0:0|Unknown|Unknown\n" if args[0] == "sacct" else ""
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

        namespace = {"PAYLOAD": {"assignments": [], "arrays": ["100"]}}
        with patch("subprocess.run", side_effect=fake_command), patch("builtins.print"):
            exec(REMOTE_READER, namespace)
        accounting_args = next(args for args in calls if args[0] == "sacct")
        self.assertIn("--array", accounting_args)
        self.assertIn("--starttime=1970-01-01", accounting_args)
        remote = {"time_utc": "now", "queue": namespace["queue"],
                  "accounting": namespace["accounting"], "runs": [{"index": 3, "summary": {}}]}
        report = combine_host("h100", self.document["assignments"][1:], remote)
        self.assertEqual(report["counts"], {"pending": 1})

    def test_real_queue_and_progress_and_overbudget_are_distinct(self):
        remote = {"time_utc": "2026-10-20T00:00:00Z",
                  "queue": {"returncode": 0, "stdout": "100_3|RUNNING|00:10|node|job\n", "stderr": ""},
                  "accounting": {"returncode": 0, "stdout": "100_3|RUNNING|00:10|0:0|start|Unknown\n", "stderr": ""},
                  "runs": [{"index": 3, "summary": {}, "last_progress": {"step": 200},
                            "last_train": {"step": 200, "tokens_seen": 3276800}}]}
        rows = self.document["assignments"][1:]
        report = combine_host("h100", rows, remote)
        self.assertEqual(report["counts"], {"running": 1})
        self.assertEqual(report["runs"][0]["training_tokens_B"], .0032768)
        remote["runs"][0]["summary"] = {"status": "stopped_over_budget", "step": 147609, "tokens_seen": 2418425856}
        report = combine_host("h100", rows, remote)
        self.assertEqual(report["counts"], {"stopped_over_budget": 1})

    def test_slurm_completed_without_summary_not_success(self):
        remote = {"time_utc": "now", "queue": {"returncode": 0, "stdout": ""},
                  "accounting": {"returncode": 0, "stdout": "100_3|COMPLETED|1:00|0:0|start|end\n"},
                  "runs": [{"index": 3, "summary": {}}]}
        report = combine_host("h100", self.document["assignments"][1:], remote)
        self.assertEqual(report["counts"], {"finished_without_summary": 1})

    def test_missing_scheduler_information_not_pending(self):
        remote = {"time_utc": "now", "queue": {"returncode": -1, "stdout": "", "stderr": "failed"},
                  "accounting": {"returncode": -1, "stdout": "", "stderr": "failed"},
                  "runs": [{"index": 3, "summary": {}}]}
        report = combine_host("h100", self.document["assignments"][1:], remote)
        self.assertEqual(report["counts"], {"unknown": 1})
        self.assertEqual(len(report["errors"]), 2)

    def test_gate_is_queried_and_state_read_without_counting_as_training(self):
        calls = []

        def fake_command(args, **kwargs):
            calls.append(args)
            output = ("100_3|PENDING|00:00|Resources|training\n"
                      "9638|RUNNING|01:00|node|stage_gate\n"
                      "9999|RUNNING|01:00|node|unrelated\n") if args[0] == "squeue" else (
                      "100_3|PENDING|00:00|0:0|Unknown|Unknown\n"
                      "9638|RUNNING|01:00|0:0|start|Unknown\n")
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

        namespace = {"PAYLOAD": {"assignments": [], "arrays": ["100"],
                                 "gates": {"p02_to_p005": "9638"}}}
        state = {"phase": "p02_to_p005", "status": "waiting", "last_checked_at": "now"}
        with patch("subprocess.run", side_effect=fake_command), patch("builtins.print"), \
                patch("pathlib.Path.exists", return_value=True), \
                patch("pathlib.Path.read_text", return_value=json.dumps(state)) as read:
            exec(REMOTE_READER, namespace)
        self.assertEqual(read.call_count, 1)
        accounting = next(args for args in calls if args[0] == "sacct")
        self.assertEqual(accounting[accounting.index("-j") + 1], "100,9638")
        self.assertNotIn("9999", namespace["queue"]["stdout"])
        remote = {"time_utc": "now", "queue": namespace["queue"], "accounting": namespace["accounting"],
                  "runs": [{"index": 3, "summary": {}}], "gates": namespace["gates"]}
        report = combine_host("h100", self.document["assignments"][1:], remote)
        self.assertEqual(report["counts"], {"pending": 1})
        self.assertEqual(report["gate_counts"], {"running": 1})
        self.assertEqual(report["gates"][0]["gate_status"], "waiting")
        self.assertFalse(report["gates"][0]["release_confirmed"])
        self.assertEqual(report["errors"], [])

    def test_failed_gate_surfaces_error_with_held_training_still_pending(self):
        remote = {"time_utc": "now",
                  "queue": {"returncode": 0, "stdout": "100_3|PENDING|00:00|JobHeldUser|training\n"},
                  "accounting": {"returncode": 0, "stdout": "9638|FAILED|01:00|1:0|start|end\n"},
                  "runs": [{"index": 3, "summary": {}}],
                  "gates": [{"phase": "p02_to_p005", "job_id": "9638", "errors": [],
                             "state": {"phase": "p02_to_p005", "status": "waiting"}}]}
        report = combine_host("h100", self.document["assignments"][1:], remote)
        self.assertEqual(report["counts"], {"pending": 1})
        self.assertEqual(report["gate_counts"], {"failed": 1})
        self.assertTrue(any("CPU gate scheduler status: failed" in error for error in report["errors"]))

    def test_completed_gate_requires_valid_released_state(self):
        remote = {"time_utc": "now", "queue": {"returncode": 0, "stdout": ""},
                  "accounting": {"returncode": 0, "stdout": "9638|COMPLETED|01:00|0:0|start|end\n"},
                  "runs": [],
                  "gates": [{"phase": "p02_to_p005", "job_id": "9638", "errors": ["invalid JSON"], "state": {}}]}
        report = combine_host("h100", [], remote)
        self.assertFalse(report["gates"][0]["release_confirmed"])
        self.assertEqual(report["gate_counts"], {"completed_without_release_confirmation": 1})
        remote["gates"][0].update(errors=[], state={"phase": "p02_to_p005", "status": "released"})
        report = combine_host("h100", [], remote)
        self.assertTrue(report["gates"][0]["release_confirmed"])
        self.assertEqual(report["counts"], {})
        self.assertEqual(report["gate_counts"], {"completed": 1})

    def test_gate_state_read_failure_is_explicit(self):
        namespace = {"PAYLOAD": {"assignments": [], "arrays": [],
                                 "gates": {"p02_to_p005": "9638"}}}
        response = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("subprocess.run", return_value=response), patch("builtins.print"), \
                patch("pathlib.Path.exists", return_value=True), \
                patch("pathlib.Path.read_text", side_effect=PermissionError("unreadable")):
            exec(REMOTE_READER, namespace)
        self.assertEqual(namespace["gates"][0]["state"], {})
        self.assertIn("unreadable", namespace["gates"][0]["errors"][0])

    def test_offline_gate_is_unknown_and_separate(self):
        self.document["stage_gate_jobs_h100"] = {"p02_to_p005": 9638}
        with patch("scripts.status_sparsedrop_clusters.subprocess.run", side_effect=OSError("offline")):
            result = query_host("h100", self.document)
        self.assertEqual(result["counts"], {"unknown": 1})
        self.assertEqual(result["gate_counts"], {"unknown": 1})
        self.assertFalse(result["gates"][0]["release_confirmed"])

    def test_pull_processes_both_hosts_without_consuming_caller_stdin(self):
        project = self.root / "mock project"
        cluster = project / "cluster"
        cluster.mkdir(parents=True)
        source = Path(__file__).resolve().parents[1]
        script = cluster / "pull_results_long.sh"
        shutil.copyfile(source / "cluster/pull_results_long.sh", script)
        (project / "scripts").mkdir()
        shutil.copyfile(source / "scripts/pull_records.py", project / "scripts/pull_records.py")
        fakebin = self.root / "fakebin"
        fakebin.mkdir()
        (fakebin / "python").symlink_to(sys.executable)
        calls = self.root / "calls.jsonl"
        wrapper = f"""#!{sys.executable}
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
value = sys.stdin.read()
with open(os.environ["TEST_CALL_LOG"], "a") as out:
    out.write(json.dumps({{"name": name, "args": sys.argv[1:], "stdin": value}}) + "\\n")
if name == "ssh":
    if os.environ.get("FAIL_HOST") in sys.argv[1:]:
        raise SystemExit(1)
    print("example queue")
"""
        for name in ("ssh", "rsync"):
            path = fakebin / name
            path.write_text(wrapper)
            path.chmod(0o755)
        environment = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
                           TEST_CALL_LOG=str(calls), SPARSEDROP_REMOTE_ROOT="/shared/project")
        completed = subprocess.run(["bash", str(script)], env=environment,
                                   input="caller input must not reach SSH", capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        recorded = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertTrue(all(row["stdin"] == "" for row in recorded))
        for host in ("h100", "h200"):
            self.assertTrue(any(row["name"] == "ssh" and host in row["args"]
                                and row["args"][-1].startswith("cd /shared/project &&") for row in recorded))
            self.assertTrue(any(row["name"] == "rsync" and "--exclude=*.pt*" in row["args"]
                                and any(arg.startswith(host + ":") for arg in row["args"]) for row in recorded))
            self.assertTrue((project / "outputs/downloaded_records/80m" / host / "queue.txt").exists())
        failed = subprocess.run(["bash", str(script)], env={**environment, "FAIL_HOST": "h200"},
                                capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertNotIn("Saved records under", failed.stdout)

    def test_pull_rejects_unsafe_remote_path_before_connecting(self):
        from scripts import pull_records
        args = ["pull_records.py", "--host", "example", "--study", "80m",
                "--remote-root", "/shared/project;echo unexpected"]
        with patch.object(sys, "argv", args), patch("subprocess.run") as run:
            with self.assertRaises(SystemExit):
                pull_records.main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
