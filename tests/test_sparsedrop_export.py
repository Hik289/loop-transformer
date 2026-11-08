"""Final sparsity contracts and backfills; standard-library fixtures, no training."""
from copy import deepcopy
import json
import math

import pytest

from scripts.export_sparsedrop_sparsity import main
from src.training.sparsedrop_export import (
    ACTIVATION_LOCATION, build_final_sparsity, export_final_sparsity,
    render_final_sparsity, validate_final_diagnostics,
)


@pytest.fixture
def final_run():
    thresholds, fractions = [0.001, 0.01, 0.1], [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    rows = [{"loop": loop, "layer": layer, "branch": branch, "tokens": 512,
             "hidden_size": 8, "residual_rms": 0.7,
             "threshold_sensitivity": [{"threshold": threshold, "near_zero_ratio": 0.2,
                                        "active_ratio": 0.8} for threshold in thresholds],
             "top_10pct_energy": 0.1,
             "topk_energy_curve": [{"fraction": fraction, "k": math.ceil(8 * fraction),
                                    "energy": fraction} for fraction in fractions]}
            for loop in (1, 2) for layer in (0, 1) for branch in ("attention", "ffn")]
    diagnostics = {"status": "ok", "protocol": {"version": "sparsedrop_eval_v1",
                   "model_mode": "eval", "activation_location": ACTIVATION_LOCATION,
                   "max_tokens": 512, "sampled_tokens": 512,
                   "activation_thresholds": thresholds, "topk_fractions": fractions},
                   "activations": rows}
    summary = {"status": "complete", "step": 100_000, "actual_steps": 100_000,
               "tokens_seen": 1_638_400_000, "actual_tokens_seen": 1_638_400_000,
               "compute_tokens_seen": 1_638_400_000, "effective_target_steps": 100_000,
               "effective_target_tokens": 1_638_400_000, "completion_reason": "stop_after_steps",
               "original_token_budget": 20_000_000_000, "overshoot_tokens": 0,
               "overshoot_steps": 0, "token_budget_completed": False,
               "final_full_validation": {"tokens": 2048, "diagnostics": diagnostics}}
    config = {"arguments": {"loops": 2, "num_layers": 2, "hidden_size": 8}}
    return summary, config


def diagnostics(final_run):
    return final_run[0]["final_full_validation"]["diagnostics"]


def check(final_run):
    return validate_final_diagnostics(diagnostics(final_run), loops=2, num_layers=2, hidden_size=8)


def test_missing_layer_is_rejected_even_if_remaining_rows_are_consistent(final_run):
    diagnostics(final_run)["activations"] = [row for row in diagnostics(final_run)["activations"] if row["layer"] == 0]
    with pytest.raises(ValueError, match="Incomplete final activation coverage"):
        check(final_run)


def test_duplicate_replacing_missing_key_is_rejected_at_same_row_count(final_run):
    rows = diagnostics(final_run)["activations"]
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="Duplicate activation key"):
        check(final_run)


@pytest.mark.parametrize("field,value,message", [
    ("tokens", 511, "sampled_tokens"), ("tokens", True, "integer"),
    ("hidden_size", 16, "configured hidden_size"), ("layer", 2, "coverage"),
    ("loop", 3, "coverage"), ("residual_rms", float("nan"), "finite"),
    ("residual_rms", -1, "valid range"), ("top_10pct_energy", float("inf"), "finite"),
])
def test_bad_row_counts_dimensions_and_scalars_are_rejected(final_run, field, value, message):
    diagnostics(final_run)["activations"][0][field] = value
    with pytest.raises(ValueError, match=message):
        check(final_run)


@pytest.mark.parametrize("field,value,message", [
    ("near_zero_ratio", float("nan"), "finite"),
    ("active_ratio", float("inf"), "finite"),
    ("near_zero_ratio", 1.1, "valid range"),
    ("active_ratio", 0.9, "sum to one"),
])
def test_invalid_threshold_ratios_are_rejected(final_run, field, value, message):
    diagnostics(final_run)["activations"][0]["threshold_sensitivity"][0][field] = value
    with pytest.raises(ValueError, match=message):
        check(final_run)


@pytest.mark.parametrize("field,value,message", [
    ("energy", float("nan"), "finite"), ("k", 0, "integer"),
    ("k", 2, "ceil"), ("fraction", 0.05, "Duplicate|duplicate"),
])
def test_invalid_topk_curve_is_rejected(final_run, field, value, message):
    diagnostics(final_run)["activations"][0]["topk_energy_curve"][0][field] = value
    with pytest.raises(ValueError, match=message):
        check(final_run)


def test_wrong_sample_count_or_missing_threshold_is_rejected(final_run):
    diagnostics(final_run)["protocol"]["sampled_tokens"] = 513
    with pytest.raises(ValueError, match="exceeds"):
        check(final_run)
    diagnostics(final_run)["protocol"]["sampled_tokens"] = 512
    diagnostics(final_run)["activations"][0]["threshold_sensitivity"].pop()
    with pytest.raises(ValueError, match="threshold coverage"):
        check(final_run)


def write_source(directory, final_run):
    directory.mkdir(parents=True)
    for name, contents in zip(("summary.json", "config.json"), final_run):
        (directory / name).write_text(json.dumps(contents), encoding="utf-8")


def test_export_preserves_source_counts_and_every_metric_and_uses_b_units(tmp_path, final_run):
    run_dir = tmp_path / "run"
    write_source(run_dir, final_run)
    before = (run_dir / "summary.json").read_bytes()
    payload = export_final_sparsity(run_dir)
    assert payload["tokens_seen"] == 1_638_400_000
    assert payload["training_tokens_B"] == 1.6384
    assert payload["original_token_budget"] == 20_000_000_000
    assert payload["original_token_budget_B"] == 20.0
    assert payload["actual_steps"] == 100_000
    assert payload["status"] == "complete"
    assert payload["coverage"]["observed_rows"] == 8
    assert payload["activations"] == diagnostics(final_run)["activations"]
    assert (run_dir / "summary.json").read_bytes() == before
    assert json.loads((run_dir / "sparsity_final.json").read_text()) == payload
    markdown = (run_dir / "README_sparsity_final.md").read_text()
    assert "1.638400 B" in markdown and "512 个有效 positions" in markdown
    assert "1,638,400,000" not in markdown
    assert markdown.count("| attention |") == 4 and markdown.count("| ffn |") == 4
    assert "2048" not in markdown  # Full-validation population is not the diagnostic sample.


def test_invalid_reexport_does_not_overwrite_valid_artifacts(tmp_path, final_run):
    run_dir = tmp_path / "run"
    write_source(run_dir, final_run)
    export_final_sparsity(run_dir)
    before = [(run_dir / name).read_bytes() for name in ("sparsity_final.json", "README_sparsity_final.md")]
    diagnostics(final_run)["activations"].pop()
    with pytest.raises(ValueError, match="coverage"):
        export_final_sparsity(run_dir, summary=final_run[0], config=final_run[1])
    assert before == [(run_dir / name).read_bytes() for name in ("sparsity_final.json", "README_sparsity_final.md")]


def test_over_budget_status_is_preserved_and_explicitly_labeled(final_run):
    summary, config = final_run
    summary.update(status="stopped_over_budget", step=147_609, actual_steps=147_609,
                   tokens_seen=2_418_425_856, actual_tokens_seen=2_418_425_856,
                   overshoot_steps=47_609, overshoot_tokens=780_025_856)
    payload = build_final_sparsity(summary, config, run_name="over-budget")
    assert payload["status"] == "stopped_over_budget"
    assert payload["actual_steps"] == 147_609
    assert payload["budget_exceeded"]
    assert payload["overshoot_tokens"] == 780_025_856
    assert "超出当前预算" in render_final_sparsity(payload)


def test_summary_actual_counts_cannot_disagree(final_run):
    final_run[0]["actual_steps"] += 1
    with pytest.raises(ValueError, match="disagree"):
        build_final_sparsity(*final_run, run_name="bad")


def test_cli_skips_unfinished_and_archives_and_strict_rejects_invalid_summary(tmp_path, final_run, capsys):
    write_source(tmp_path / "valid", final_run)
    (tmp_path / "unfinished").mkdir()
    write_source(tmp_path / "archives", final_run)
    write_source(tmp_path / "archives" / "old-run", final_run)
    bad = deepcopy(final_run)
    bad[0]["final_full_validation"]["diagnostics"]["activations"].pop()
    write_source(tmp_path / "invalid", bad)
    assert main([str(tmp_path), "--strict"]) == 1
    output = capsys.readouterr()
    assert '"exported": 1' in output.out
    assert '"skipped_without_summary": 1' in output.out
    assert '"failed": 1' in output.out
    assert "coverage" in output.err
    assert (tmp_path / "valid" / "sparsity_final.json").is_file()
    assert not list((tmp_path / "archives").rglob("sparsity_final.json"))
    assert main([str(tmp_path)]) == 0  # Best-effort mode still reports the failure.


def test_cli_strict_allows_normal_unfinished_runs(tmp_path, final_run):
    write_source(tmp_path / "finished", final_run)
    (tmp_path / "unfinished").mkdir()
    assert main([str(tmp_path), "--strict"]) == 0
