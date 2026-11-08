"""Final-only reporting, budget separation and serialized concurrent refreshes."""
from copy import deepcopy
import fcntl
import json
import math
import multiprocessing
from pathlib import Path

import pytest

from scripts.update_sparsedrop_readme import main
from src.training import sparsedrop_readme as report
from src.training.sparsedrop_export import ACTIVATION_LOCATION


PREFIX = "# Original research\n\n## 4. Measurements\nKeep equations $x^2$.\n\n"
SUFFIX = "## 6. Theory\n\nKeep theory, citations [1], and figures exactly.\n\n## 7. Records\nUnchanged.\n"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path):
    config_dir = tmp_path / "configs"
    entries = []
    conditions = [(1, "none", 0), (4, "both", .1), (8, "ffn", .1)]
    for index, (loops, mode, p) in enumerate(conditions):
        name = f"h{loops}_{mode}_p{p:.2f}_constant_d0.00_s42"
        config = {"loops": loops, "mode": mode, "p": p, "schedule": "constant", "delta": 0., "seed": 42,
                  "batch_size": 32, "grad_accum": 1, "seq_len": 512, "max_steps": 1220704,
                  "stop_after_steps": 100000, "token_budget": 20_000_000_000,
                  "output_dir": f"/remote/must/not/be/read/{name}", "num_layers": 2, "hidden_size": 8}
        _write(config_dir / f"{index}.json", config)
        entries.append({"index": index, "name": name, "config": f"configs/{index}.json", "seed": 42})
    _write(config_dir / "manifest.json", entries)
    readme = tmp_path / "original README.md"
    readme.write_text(PREFIX + "## 5. Results\n\nOld duplicate and merged rows.\n\n" + SUFFIX, encoding="utf-8")
    return {"manifest_path": config_dir / "manifest.json", "results_root": tmp_path / "local results", "readme_path": readme}


def _final(workspace, index=0, *, step=100000, status="complete", loss=2.0):
    entry = json.loads(workspace["manifest_path"].read_text())[index]
    args = json.loads((workspace["manifest_path"].parent / f"{index}.json").read_text())
    config = {"arguments": args, "model": {"num_layers": 2, "hidden_size": 8},
              "data": {"validation": {"tokens": 2049, "chunks": 4, "prediction_tokens": 2048,
                                      "trailing_prediction_tokens_excluded": 0}}}
    thresholds, fractions = [.001, .01, .1], [.1, 1.]
    activations = []
    for loop in range(1, args["loops"] + 1):
        for layer in range(2):
            for branch in ("attention", "ffn"):
                ratio = .2 if branch == "ffn" else .4
                activations.append({"loop": loop, "layer": layer, "branch": branch, "tokens": 512, "hidden_size": 8,
                                    "residual_rms": .7, "top_10pct_energy": .1,
                                    "threshold_sensitivity": [{"threshold": t, "near_zero_ratio": ratio, "active_ratio": 1 - ratio} for t in thresholds],
                                    "topk_energy_curve": [{"fraction": f, "k": math.ceil(8 * f), "energy": f} for f in fractions]})
    diag = {"status": "ok", "protocol": {"model_mode": "eval", "activation_location": ACTIVATION_LOCATION,
            "max_tokens": 512, "sampled_tokens": 512, "activation_thresholds": thresholds, "topk_fractions": fractions},
            "activations": activations, "loop_states": [{"loop": loop, "update_mean_l2": loop * 2., "update_rms": 9999.} for loop in range(1, args["loops"] + 1)],
            "cross_loop": {"status": "ok" if args["loops"] > 1 else "not_applicable_H1",
                           "loop_indices": list(range(1, args["loops"] + 1)),
                           "mean_update_cosine": .1234 if args["loops"] > 1 else None,
                           "update_cosine": [[1. if i == j else .1234 for j in range(args["loops"])] for i in range(args["loops"])]}}
    summary = {"status": status, "step": step, "actual_steps": step, "tokens_seen": step * 16384,
               "actual_tokens_seen": step * 16384, "effective_target_steps": 100000, "effective_target_tokens": 1638400000,
               "mean_train_tokens_per_second": 250000., "peak_gpu_memory_bytes": 2 * 1024 ** 3,
               "final_full_validation": {"loss": loss, "perplexity": math.exp(loss), "tokens": 2048, "chunks": 4,
                                         "evaluation_scope": "all_complete_validation_chunks", "trailing_prediction_tokens_excluded": 0,
                                         "diagnostics": diag}}
    run = workspace["results_root"] / entry["name"]
    _write(run / "config.json", config)
    _write(run / "summary.json", summary)
    train_step = step - step % 100
    (run / "train.jsonl").write_text(json.dumps({"event": "train", "step": train_step, "tokens_seen": train_step * 16384,
        "loss": 99.987654, "grad_norm_preclip": 1.25, "grad_norm_preclip_last": 88.25}) + "\n", encoding="utf-8")
    return run, summary, config


def test_unfinished_never_uses_intermediate_quality_or_archive(workspace):
    run, summary, _ = _final(workspace)
    (run / "summary.json").unlink()
    _write(run / "archive" / "summary.json", summary)
    (run / "eval.jsonl").write_text(json.dumps({"event": "validation_fixed", "loss": 1.111111, "perplexity": 3.0377}) + "\n")
    result = report.update_readme_results(**workspace)
    assert result["counts"] == {"total": 3, "finished_valid_exact": 0, "overbudget": 0, "unfinished": 3, "invalid": 0, "budget_mismatch": 0}
    assert result["rows"][0]["metrics"] is None
    text = workspace["readme_path"].read_text()
    assert "99.987654" not in text and "1.111111" not in text and "7.389056" not in text
    assert "100,000" in text and "1.638400" in text


def test_overbudget_actual_final_and_historical_fixed_are_separate(workspace):
    run, summary, _ = _final(workspace, 1, step=147609, status="stopped_over_budget", loss=3.)
    fixed = {"event": "validation_fixed", "step": 100000, "tokens_seen": 1638400000,
             "tokens": 65536, "loss": 4., "perplexity": math.exp(4.)}
    (run / "eval.jsonl").write_text(json.dumps(fixed) + "\n")
    result = report.update_readme_results(**workspace)
    assert result["counts"]["overbudget"] == 1 and result["counts"]["finished_valid_exact"] == 0
    row = result["rows"][1]
    assert row["actual_steps"] == 147609 and row["metrics"]["loss"] == 3.
    assert row["metrics"]["gradient_window_end_step"] == 147600
    assert row["metrics"]["ffn_near_zero_eps0.01"] == pytest.approx(.2)
    assert row["metrics"]["attention_near_zero_eps0.01"] == pytest.approx(.4)
    assert row["metrics"]["mean_update_l2"] == 5.
    text = workspace["readme_path"].read_text()
    main_tables, excluded = text.split("### 独立表：", 1)
    assert "20.085537" not in main_tables and "54.598150" not in main_tables
    assert "147,609" in main_tables and "2.418426" in main_tables and "不纳入主表" in main_tables
    assert "20.085537" in excluded and "54.598150" in excluded
    assert "9999.000000" not in text


def test_incremental_second_completion_preserves_first_and_idempotence(workspace):
    first_run, _, _ = _final(workspace)
    first = report.update_readme_results(**workspace)
    assert first["counts"]["finished_valid_exact"] == 1
    _final(workspace, 2, loss=2.5)
    second = report.update_readme_results(**workspace)
    assert second["counts"]["finished_valid_exact"] == 2
    assert second["rows"][0]["metrics"] == first["rows"][0]["metrics"]
    assert second["rows"][0]["metrics"]["mean_update_cosine"] is None
    assert second["rows"][0]["metrics"]["gradient_norm_last_window"] == 1.25
    assert second["rows"][0]["metrics"]["mean_train_B_per_second"] == .00025
    assert second["rows"][0]["metrics"]["peak_memory_GiB"] == 2.
    outputs = [Path(second[k]) for k in ("readme_path", "live_readme_path", "status_path")]
    before = [p.read_bytes() for p in outputs]
    report.update_readme_results(**workspace)
    assert [p.read_bytes() for p in outputs] == before
    text = outputs[0].read_text()
    assert text.startswith(PREFIX) and text.endswith(SUFFIX)
    assert text.count(report.BEGIN) == 1 and text.count(report.END) == 1
    live = outputs[1].read_text()
    assert "local results/" in live and "](<" in live
    assert "/remote/" not in live and str(workspace["results_root"]) not in live
    assert all(not Path(p).is_absolute() for p in second["rows"][0]["sources"].values())
    assert "N/A" not in live  # no cosine column in the compact all-condition table
    assert first_run.is_dir()


@pytest.mark.parametrize("corruption", ["missing_activation", "nan_loss", "wrong_ppl", "wrong_actual", "under_target", "wrong_condition", "invalid_json"])
def test_invalid_final_rejected_without_partial_metrics(workspace, corruption):
    run, summary, config = _final(workspace)
    if corruption == "missing_activation":
        summary["final_full_validation"]["diagnostics"]["activations"].pop()
    elif corruption == "nan_loss":
        summary["final_full_validation"]["loss"] = float("nan")
    elif corruption == "wrong_ppl":
        summary["final_full_validation"]["perplexity"] = 20.
    elif corruption == "wrong_actual":
        summary["actual_steps"] += 1
    elif corruption == "under_target":
        summary.update(step=99999, actual_steps=99999, tokens_seen=99999 * 16384, actual_tokens_seen=99999 * 16384)
    elif corruption == "wrong_condition":
        config["arguments"]["mode"] = "ffn"
        _write(run / "config.json", config)
    _write(run / "summary.json", summary)
    if corruption == "invalid_json":
        (run / "summary.json").write_text("{incomplete")
    result = report.update_readme_results(**workspace)
    assert result["counts"]["invalid"] == 1
    assert result["rows"][0]["metrics"] is None and result["rows"][0]["errors"]
    assert "7.389056" not in workspace["readme_path"].read_text()


def test_summary_declaring_another_budget_cannot_enter_main_tables(workspace):
    run, summary, _ = _final(workspace)
    summary["effective_target_steps"] = 200000
    summary["effective_target_tokens"] *= 2
    _write(run / "summary.json", summary)
    result = report.update_readme_results(**workspace)
    assert result["counts"]["invalid"] == 1
    assert result["counts"]["finished_valid_exact"] == 0
    assert "7.389056" not in workspace["readme_path"].read_text().split("### 独立表：")[0]


@pytest.mark.parametrize("corruption", ["fixed_scope", "fixed_tokens", "data_signature", "config_target", "config_actual",
                                       "wrong_batch", "missing_states", "null_states", "bad_state_type", "missing_cross",
                                       "null_cross", "missing_cross_mean", "bad_cross_matrix"])
def test_full_scope_config_consistency_and_complete_diagnostics_required(workspace, corruption):
    run, summary, config = _final(workspace, 1)
    final = summary["final_full_validation"]
    diagnostics = final["diagnostics"]
    if corruption == "fixed_scope":
        final["evaluation_scope"] = "fixed_subset"
    elif corruption == "fixed_tokens":
        final.update(tokens=1024, chunks=2)
    elif corruption == "data_signature":
        config["data"]["validation"]["prediction_tokens"] = 1024
    elif corruption == "config_target":
        config.update(effective_target_steps=200000, effective_target_tokens=3276800000)
    elif corruption == "config_actual":
        config["actual_steps"] = 90000
    elif corruption == "wrong_batch":
        config["arguments"]["batch_size"] = 16
    elif corruption == "missing_states":
        del diagnostics["loop_states"]
    elif corruption == "null_states":
        diagnostics["loop_states"] = None
    elif corruption == "bad_state_type":
        diagnostics["loop_states"][0] = None
    elif corruption == "missing_cross":
        del diagnostics["cross_loop"]
    elif corruption == "null_cross":
        diagnostics["cross_loop"] = None
    elif corruption == "missing_cross_mean":
        del diagnostics["cross_loop"]["mean_update_cosine"]
    elif corruption == "bad_cross_matrix":
        diagnostics["cross_loop"]["update_cosine"][0].pop()
    _write(run / "summary.json", summary)
    _write(run / "config.json", config)
    result = report.update_readme_results(**workspace)
    assert result["counts"]["invalid"] == 1
    assert result["rows"][1]["metrics"] is None


def test_all_52_manifest_conditions_expand_to_correct_fixed_tables():
    root = Path(__file__).resolve().parents[1]
    entries = report.load_manifest(root / "configs/sparsedrop_long/manifest.json")
    assert len(entries) == 52
    fixed = [r for r in entries if r["schedule"] == "constant"]
    assert len(fixed) == 40
    assert len([r for r in fixed if r["loops"] == 4 and r["p"] in (0, .1)]) == 4
    assert len([r for r in entries if r["loops"] == 8 and r["p"] == .1]) == 15
    assert len([r for r in fixed if r["loops"] in (4, 8)]) == 20
    assert {(r["effective_target_steps"], r["effective_target_tokens"]) for r in entries} == {(100000, 1638400000)}


def _refresh_child(workspace, entered_scan, result_queue):
    original_scan = report.scan_results
    def observed_scan(*args):
        entered_scan.set()
        return original_scan(*args)
    report.scan_results = observed_scan
    try:
        result_queue.put(report.update_readme_results(**workspace)["counts"])
    except Exception as error:
        result_queue.put({"error": repr(error)})


def test_lock_is_acquired_before_scan_and_waiter_sees_new_completion(workspace):
    _final(workspace, 0)
    context = multiprocessing.get_context("fork")
    entered_scan, queue = context.Event(), context.Queue()
    lock_path = workspace["readme_path"].with_name(workspace["readme_path"].name + ".sparsedrop.lock")
    child = context.Process(target=_refresh_child, args=(workspace, entered_scan, queue))
    # Child opens a separate lock descriptor. Parent explicitly unlocks its
    # inherited open-file description before waiting for the child.
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        child.start()
        try:
            assert not entered_scan.wait(.25), "Source scan happened before acquiring the report lock"
            _final(workspace, 2, loss=2.5)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    child.join(10)
    if child.is_alive():
        child.terminate()
        child.join(5)
        pytest.fail("Concurrent refresher did not finish after lock release")
    counts = queue.get(timeout=2)
    assert child.exitcode == 0 and counts["finished_valid_exact"] == 2
    assert entered_scan.is_set()


def test_cli_strict_reports_errors_but_valid_run_still_published(workspace, capsys):
    _final(workspace, 0)
    run, summary, _ = _final(workspace, 1)
    summary["final_full_validation"]["diagnostics"]["activations"].pop()
    _write(run / "summary.json", summary)
    arguments = ["--manifest", str(workspace["manifest_path"]), "--results-root", str(workspace["results_root"]),
                 "--readme", str(workspace["readme_path"]), "--strict"]
    assert main(arguments) == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["counts"]["finished_valid_exact"] == 1
    assert "coverage" in output.err
    assert "7.389056" in workspace["readme_path"].read_text()


def test_invalid_section_layout_does_not_replace_original_or_outputs(workspace):
    original = "# Wrong document without numbered sections\n"
    workspace["readme_path"].write_text(original)
    with pytest.raises(ValueError, match="section 5"):
        report.update_readme_results(**workspace)
    assert workspace["readme_path"].read_text() == original
    assert not workspace["readme_path"].with_name("sparsedrop_readme_status.json").exists()
