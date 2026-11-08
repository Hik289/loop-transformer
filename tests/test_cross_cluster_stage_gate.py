"""Run with python -m unittest tests.test_cross_cluster_stage_gate (no torch)."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.cross_cluster_stage_gate import (
    GateError, RetryableError, STAGES, StageGate, accounting_rows, array_task_ids,
    control_record, validate_plan,
)


def example_plan():
    plan = {"schema_version": 1, "phase": "p02_to_p005", "owner": "researcher"}
    for key, stage in (("requirements", "p02"), ("release_targets", "p005")):
        indices = sorted(STAGES[stage])
        plan[key] = [
            {"cluster": "h200cluster", "job_ids": [f"53342_{i}" for i in indices[:6]]},
            {"cluster": "h100cluster", "job_ids": [f"60000_{i}" for i in indices[6:]]},
        ]
    return plan


class FakeSlurm:
    def __init__(self, plan):
        self.plan = plan
        self.prerequisites = {
            (group["cluster"], job_id): ("COMPLETED", "0:0")
            for group in plan["requirements"] for job_id in group["job_ids"]}
        self.targets = {
            (group["cluster"], job_id): ("PENDING", "JobHeldUser")
            for group in plan["release_targets"] for job_id in group["job_ids"]}
        self.commands, self.released = [], []
        self.fail_release_number = None
        self.fail_after_apply = False
        self.transport_down = False
        self.wrong_owner = False
        self.local_cluster = "h100cluster"

    def __call__(self, argv):
        self.commands.append(argv)
        if self.transport_down:
            raise RetryableError("controller timeout")
        if argv == ["scontrol", "show", "config"]:
            return f"ClusterName             = {self.local_cluster}\nSlurmctldHost = h100server\n"
        cluster = argv[argv.index("-M") + 1] if "-M" in argv else self.local_cluster
        if argv[0] == "sacct":
            ids = argv[argv.index("-j") + 1].split(",")
            records = []
            for job_id in ids:
                item = self.prerequisites.get((cluster, job_id))
                if item is not None:
                    state, code = item
                    owner = "wrong" if self.wrong_owner else "researcher"
                    records.append(f"{cluster}|{job_id}|{owner}|{state}|{code}")
            return "\n".join(records)
        job_id = argv[-1]
        key = cluster, job_id
        if "release" in argv:
            number = len(self.released) + 1
            if self.fail_release_number == number and not self.fail_after_apply:
                self.fail_release_number = None
                raise RetryableError("release transport failure before application")
            self.released.append(key)
            self.targets[key] = "PENDING", "Resources"
            if self.fail_release_number == number and self.fail_after_apply:
                self.fail_release_number = None
                raise RetryableError("lost reply after release application")
            return ""
        state, reason = self.targets[key]
        array, task = job_id.split("_")
        return (f"JobId=123456 ArrayJobId={array} ArrayTaskId={task} "
                f"UserId=researcher(1000) JobState={state} Reason={reason} "
                f"Dependency=(null) ExitCode=0:0")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_path = Path(self.tmp.name) / "gate.state.json"
        self.plan = example_plan()
        self.slurm = FakeSlurm(self.plan)

    def gate(self):
        return StageGate(self.plan, self.state_path, self.slurm)

    def test_releases_all_only_after_both_clusters_complete(self):
        gate = self.gate()
        self.assertTrue(gate.tick())
        self.assertEqual(len(self.slurm.released), 12)
        self.assertEqual(gate.state["status"], "released")
        self.assertTrue(gate.tick())
        self.assertEqual(len(self.slurm.released), 12)

    def test_missing_accounting_is_not_completion(self):
        self.slurm.prerequisites.pop(next(iter(self.slurm.prerequisites)))
        self.assertFalse(self.gate().tick())
        self.assertEqual(self.slurm.released, [])

    def test_sacct_always_requests_explicit_array_expansion(self):
        self.assertTrue(self.gate().tick())
        queries = [argv for argv in self.slurm.commands if argv[0] == "sacct"]
        self.assertEqual(len(queries), 2)
        self.assertTrue(all("--array" in argv for argv in queries))

    def test_compressed_accounting_is_still_rejected(self):
        with self.assertRaises(GateError):
            accounting_rows("h200cluster|53342_[11-13]|researcher|PENDING|0:0",
                            "h200cluster", {"53342_11", "53342_12", "53342_13"}, "researcher")

    def test_active_prerequisite_keeps_targets_held(self):
        key = next(iter(self.slurm.prerequisites))
        for state in ("PENDING", "RUNNING", "COMPLETING", "REQUEUED"):
            with self.subTest(state=state):
                self.slurm.prerequisites[key] = state, "0:0"
                self.assertFalse(self.gate().tick())
                self.assertEqual(self.slurm.released, [])

    def test_failed_and_nonzero_completed_fail_closed(self):
        key = next(iter(self.slurm.prerequisites))
        for state, code in (("FAILED", "1:0"), ("CANCELLED by 1000", "0:15"),
                            ("TIMEOUT", "0:9"), ("COMPLETED", "1:0"),
                            ("UNKNOWN", "0:0")):
            with self.subTest(state=state):
                self.slurm.prerequisites[key] = state, code
                with self.assertRaises(GateError):
                    self.gate().tick()
                self.assertEqual(self.slurm.released, [])

    def test_transport_failure_does_not_release(self):
        self.slurm.transport_down = True
        with self.assertRaises(RetryableError):
            self.gate().tick()
        self.assertEqual(self.slurm.released, [])

    def test_wrong_owner_fails_closed(self):
        self.slurm.wrong_owner = True
        with self.assertRaises(GateError):
            self.gate().tick()
        self.assertEqual(self.slurm.released, [])

    def test_all_targets_are_checked_before_any_release(self):
        key = list(self.slurm.targets)[-1]
        self.slurm.targets[key] = "PENDING", "JobHeldAdmin"
        with self.assertRaises(GateError):
            self.gate().tick()
        self.assertEqual(self.slurm.released, [])

    def test_unexpected_unheld_pending_without_intent_fails_closed(self):
        self.slurm.targets[next(iter(self.slurm.targets))] = "PENDING", "Resources"
        with self.assertRaises(GateError):
            self.gate().tick()
        self.assertEqual(self.slurm.released, [])

    def test_partial_release_retries_without_releasing_twice(self):
        self.slurm.fail_release_number = 4
        with self.assertRaises(RetryableError):
            self.gate().tick()
        self.assertEqual(len(self.slurm.released), 3)
        self.assertTrue(self.gate().tick())
        self.assertEqual(len(self.slurm.released), 12)
        self.assertEqual(len(set(self.slurm.released)), 12)

    def test_lost_success_reply_is_safe_to_retry(self):
        self.slurm.fail_release_number = 4
        self.slurm.fail_after_apply = True
        with self.assertRaises(RetryableError):
            self.gate().tick()
        self.assertEqual(len(self.slurm.released), 4)
        self.assertTrue(self.gate().tick())
        self.assertEqual(len(self.slurm.released), 12)
        self.assertEqual(len(set(self.slurm.released)), 12)

    def test_known_running_or_completed_targets_are_not_released_again(self):
        targets = list(self.slurm.targets)
        self.slurm.targets[targets[0]] = "RUNNING", "None"
        self.slurm.targets[targets[1]] = "COMPLETED", "None"
        self.assertTrue(self.gate().tick())
        self.assertEqual(len(self.slurm.released), 10)

    def test_changed_plan_cannot_reuse_state(self):
        self.gate()
        self.plan["requirements"][0]["job_ids"][0] = "99999_11"
        with self.assertRaises(GateError):
            self.gate()

    def test_duplicate_or_wrong_task_plan_rejected(self):
        for wrong_id in ("53342", "53342_[5-7]", "53342_0", "53342_6"):
            with self.subTest(job_id=wrong_id):
                plan = deepcopy(self.plan)
                plan["release_targets"][0]["job_ids"][0] = wrong_id
                with self.assertRaises(GateError):
                    validate_plan(plan)

    def test_accounting_wrong_cluster_extra_id_or_conflict_rejected(self):
        records = [
            "other|53342_11|researcher|COMPLETED|0:0",
            "h200cluster|53342_12|researcher|COMPLETED|0:0",
            "h200cluster|53342_11|researcher|COMPLETED|0:0\n"
            "h200cluster|53342_11|researcher|FAILED|1:0",
        ]
        for record in records:
            with self.subTest(record=record), self.assertRaises(GateError):
                accounting_rows(record, "h200cluster", {"53342_11"}, "researcher")

    def test_controller_identity_rejected(self):
        good = ("JobId=1 ArrayJobId=53342 ArrayTaskId=5 UserId=researcher(1000) "
                "JobState=PENDING Reason=JobHeldUser Dependency=(null)")
        for bad in (good.replace("TaskId=5", "TaskId=6"),
                    good.replace("researcher", "other"), good + "\n" + good,
                    good + " Cluster=wrong"):
            with self.subTest(record=bad), self.assertRaises(GateError):
                control_record(bad, "h200cluster", "53342_5", "researcher")

    def test_compressed_controller_array_includes_requested_element(self):
        good = ("JobId=53342 ArrayJobId=53342 ArrayTaskId=5,7,16-17,31-33:2%2 "
                "UserId=researcher(1000) JobState=PENDING Reason=JobHeldUser Dependency=(null)")
        for task in (5, 7, 16, 17, 31, 33):
            with self.subTest(task=task):
                control_record(good, "h200cluster", f"53342_{task}", "researcher")
        for task in (6, 15, 32):
            with self.subTest(task=task), self.assertRaises(GateError):
                control_record(good, "h200cluster", f"53342_{task}", "researcher")

    def test_malformed_compressed_controller_array_rejected(self):
        for value in ("", "5,,7", "5-", "-1", "5-1000000000", "8-5", "5-7:0",
                      "5;7", "5%0", "[5-7]", "5-7%", "1-3:"):
            with self.subTest(value=value), self.assertRaises(GateError):
                array_task_ids(value)

    def test_compressed_controller_records_only_release_explicit_elements(self):
        original = self.slurm
        group_tasks = {
            group["cluster"]: ",".join(job_id.split("_")[1] for job_id in group["job_ids"]) + "%2"
            for group in self.plan["release_targets"]}

        def shared_record(argv):
            result = original(argv)
            if argv[0] == "scontrol" and "job" in argv:
                cluster = argv[argv.index("-M") + 1] if "-M" in argv else "h100cluster"
                task = argv[-1].split("_")[1]
                result = result.replace(f"ArrayTaskId={task} ", f"ArrayTaskId={group_tasks[cluster]} ")
            return result

        self.assertTrue(StageGate(self.plan, self.state_path, shared_record).tick())
        self.assertEqual(len(self.slurm.released), 12)
        self.assertTrue(all(job_id.count("_") == 1 and "[" not in job_id
                            for _, job_id in self.slurm.released))

    def test_intent_written_before_release(self):
        original = self.slurm

        def inspect_intent(argv):
            if "release" in argv:
                state = json.loads(self.state_path.read_text())
                cluster = argv[argv.index("-M") + 1] if "-M" in argv else "h100cluster"
                self.assertIn("intent_at", state["releases"][f"{cluster}/{argv[-1]}"])
            return original(argv)

        self.assertTrue(StageGate(self.plan, self.state_path, inspect_intent).tick())

    def test_unexpected_dependency_blocks_all_releases(self):
        original = self.slurm

        def wrong_dependency(argv):
            result = original(argv)
            if argv[0] == "scontrol" and "show" in argv:
                return result.replace("Dependency=(null)", "Dependency=afterok:cancelled")
            return result

        with self.assertRaises(GateError):
            StageGate(self.plan, self.state_path, wrong_dependency).tick()
        self.assertEqual(self.slurm.released, [])

    def test_partial_release_can_wait_for_any_resource_limit(self):
        self.slurm.fail_release_number = 2
        with self.assertRaises(RetryableError):
            self.gate().tick()
        first = self.slurm.released[0]
        self.slurm.targets[first] = "PENDING", "AssocGrpMemLimit"
        self.assertTrue(self.gate().tick())
        self.assertEqual(len(self.slurm.released), 12)

    def test_second_stage_plan_is_valid(self):
        self.plan["phase"] = "p005_to_schedule"
        for key, stage in (("requirements", "p005"), ("release_targets", "schedule")):
            indices = sorted(STAGES[stage])
            for n, group in enumerate(self.plan[key]):
                group["job_ids"] = [f"{53342 + n}_{i}" for i in indices[n * 6:(n + 1) * 6]]
        self.assertEqual(validate_plan(self.plan)["phase"], "p005_to_schedule")

    def test_default_controller_must_be_verified_h100(self):
        self.slurm.local_cluster = "h200cluster"
        with self.assertRaises(GateError):
            self.gate().tick()
        self.assertEqual(self.slurm.released, [])

    def test_local_h100_control_uses_default_cross_h200_keeps_cluster(self):
        self.assertTrue(self.gate().tick())
        config_queries = [argv for argv in self.slurm.commands
                          if argv == ["scontrol", "show", "config"]]
        self.assertEqual(len(config_queries), 1)
        controls = [argv for argv in self.slurm.commands
                    if argv[0] == "scontrol" and argv != ["scontrol", "show", "config"]]
        for argv in controls:
            if argv[-1].startswith("60000_"):
                self.assertNotIn("-M", argv)
            else:
                self.assertEqual(argv[argv.index("-M") + 1], "h200cluster")
        accounting = [argv for argv in self.slurm.commands if argv[0] == "sacct"]
        self.assertTrue(all("-M" in argv for argv in accounting))

    def test_missing_or_ambiguous_default_cluster_config_fails_closed(self):
        original = self.slurm
        for output in ("", "ClusterName = h100cluster\nClusterName = h200cluster\n"):
            with self.subTest(output=output):
                def bad_config(argv):
                    if argv == ["scontrol", "show", "config"]:
                        return output
                    return original(argv)
                with self.assertRaises(GateError):
                    StageGate(self.plan, self.state_path, bad_config).tick()
                self.assertEqual(self.slurm.released, [])


if __name__ == "__main__":
    unittest.main()
