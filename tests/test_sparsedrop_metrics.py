import math

import pytest
import torch

from src.training.sparsedrop_metrics import SparseDropMetrics, linear_cka, token_cosine


def test_cosine_is_per_token_not_flattened():
    x = torch.tensor([[100.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([[100.0, 0.0], [0.0, -1.0]])
    assert token_cosine(x, y) == pytest.approx(0.0, abs=1e-11)
    assert torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0) > 0.99


def test_cka_matches_centered_feature_formula_and_translation_invariance():
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    y = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    xc, yc = x - x.mean(0), y - y.mean(0)
    reference = ((xc.T @ yc).norm().square() / ((xc.T @ xc).norm() * (yc.T @ yc).norm() + 1e-12)).item()
    assert linear_cka(x, y) == pytest.approx(reference, abs=1e-12)
    assert linear_cka(x + 100, y - 37) == pytest.approx(reference, abs=1e-12)
    assert linear_cka(x, x) == pytest.approx(1.0)


def test_residual_threshold_boundary_energy_and_h1_na():
    collector = SparseDropMetrics(max_tokens=2, thresholds=(0.0, 1.0), topk_fractions=(0.25, 1.0))
    residual = torch.tensor([[[0.0, 0.0, 3.0, 4.0], [0.0, 1.0, 0.0, 0.0]]])
    collector.begin_batch()
    collector.record_state(0, torch.zeros_like(residual))
    collector.record_activation(1, 0, "attention", residual)
    collector.record_state(1, residual)
    collector.end_batch()
    result = collector.compute()
    assert result["cross_loop"]["status"] == "not_applicable_H1"
    assert result["cross_loop"]["mean_update_cosine"] is None
    activation = result["activations"][0]
    assert activation["threshold_sensitivity"][0]["near_zero_ratio"] == 5 / 8
    assert activation["threshold_sensitivity"][1]["near_zero_ratio"] == 6 / 8
    assert activation["threshold_sensitivity"][1]["active_ratio"] == 2 / 8
    assert activation["residual_rms"] == pytest.approx(math.sqrt(26 / 8))
    assert activation["top_10pct_energy"] == pytest.approx((16 / 25 + 1) / 2)
    assert activation["topk_energy_curve"][-1]["energy"] == pytest.approx(1.0)
    assert result["loop_states"][0]["update_mean_l2"] == pytest.approx(3.0)


def _feed(collector, x, valid_mask=None):
    collector.begin_batch(valid_mask)
    collector.record_state(0, x)
    collector.record_activation(1, 0, "ffn", x)
    collector.record_state(1, 2 * x)
    collector.record_activation(2, 0, "ffn", -x)
    collector.record_state(2, x)
    collector.end_batch()


def test_same_fixed_tokens_across_batch_boundaries_and_memory_bound():
    x = torch.arange(1, 41).reshape(10, 4).float()
    full, split = SparseDropMetrics(max_tokens=5), SparseDropMetrics(max_tokens=5)
    _feed(full, x)
    _feed(split, x[:3])
    _feed(split, x[3:])
    assert full.compute() == split.compute()
    assert split.sampled_tokens == 5
    assert split.full
    assert all(sum(chunk.shape[0] for chunk in chunks) == 5 for chunks in split._state_chunks.values())
    assert all(chunk.device.type == "cpu" and not chunk.requires_grad
               for chunks in split._state_chunks.values() for chunk in chunks)


def test_mask_alignment_and_opposite_updates():
    collector = SparseDropMetrics(max_tokens=4)
    x = torch.tensor([[[1.0, 2.0], [30.0, 40.0], [2.0, 1.0]]])
    _feed(collector, x, torch.tensor([[True, False, True]]))
    result = collector.compute()
    assert result["protocol"]["sampled_tokens"] == 2
    assert result["cross_loop"]["mean_state_cosine"] == pytest.approx(1.0)
    assert result["cross_loop"]["mean_update_cosine"] == pytest.approx(-1.0)
    assert result["cross_loop"]["mean_linear_cka"] == pytest.approx(1.0)
    assert result["loop_states"][0]["update_mean_l2"] == pytest.approx(math.sqrt(5))


def test_no_valid_tokens_and_zero_energy_do_not_make_nan():
    collector = SparseDropMetrics()
    _feed(collector, torch.zeros(3, 2), torch.zeros(3, dtype=torch.bool))
    assert collector.compute()["status"] == "no_valid_tokens"
    collector = SparseDropMetrics()
    _feed(collector, torch.zeros(3, 2))
    result = collector.compute()
    assert result["cross_loop"]["mean_update_cosine"] == 0
    assert result["cross_loop"]["mean_linear_cka"] == 0
    assert result["activations"][0]["top_10pct_energy"] == 0


def test_eval_mode_and_gradient_and_duplicate_guards():
    collector = SparseDropMetrics()
    with pytest.raises(ValueError, match="eval"):
        collector.begin_batch(training=True)
    collector.begin_batch()
    with pytest.raises(ValueError, match="no_grad"):
        collector.record_state(0, torch.ones(2, 3, requires_grad=True))
    collector.record_state(0, torch.ones(2, 3))
    with pytest.raises(ValueError, match="unique"):
        collector.record_state(0, torch.ones(2, 3))


def test_incomplete_loop_state_and_batch_shape_are_rejected():
    collector = SparseDropMetrics()
    collector.begin_batch()
    collector.record_state(0, torch.ones(2, 3))
    with pytest.raises(ValueError, match="token dimensions"):
        collector.record_state(1, torch.ones(3, 3))
    collector.record_state(2, torch.ones(2, 3))
    with pytest.raises(ValueError, match="every completed loop"):
        collector.end_batch()


def test_single_token_cka_is_explicitly_unavailable():
    collector = SparseDropMetrics(max_tokens=1)
    _feed(collector, torch.ones(1, 4))
    result = collector.compute()
    assert result["cross_loop"]["linear_cka"] is None
    assert result["cross_loop"]["cka_status"] == "not_applicable_fewer_than_two_tokens"


def _aggregate_run(seed=42, p=0.0, complete=True):
    mode = "none" if p == 0 else "both"
    config = {
        "arguments": {"loops": 4, "mode": mode, "p": p, "schedule": "constant", "delta": 0,
                      "seed": seed, "data_seed": seed + 100, "max_steps": 10, "batch_size": 2, "seq_len": 8},
        "model": {"hidden_size": 4, "residual_dropout_mode": mode, "residual_dropout_p": p},
        "optimizer": {"name": "AdamW"}, "data": {"train": {"tokens": 100}},
        "data_manifest": {"tokenizer": "fixed-tokenizer", "dataset_sha256": "fixed-data"},
        "token_budget": 160, "initial_parameters_sha256": f"same-for-seed-{seed}",
    }
    from scripts.aggregate_sparsedrop import condition
    evaluation = {"event": "validation_full_final" if complete else "validation_fixed",
                  "step": 10 if complete else 5, "tokens_seen": 160 if complete else 80,
                  "tokens": 64, "chunks": 8, "loss": 4.0 - p, "perplexity": math.exp(4.0 - p)}
    return {"directory": f"run-seed{seed}-p{p}", "config": config, "condition": condition(config),
            "seed": seed, "complete": complete, "status": "complete" if complete else "in_progress",
            "evaluation": evaluation, "summary": {}, "warnings": []}


def test_aggregation_rejects_unpaired_config_and_partial_final_means():
    from scripts.aggregate_sparsedrop import aggregate, pairing_errors
    baseline, treatment = _aggregate_run(), _aggregate_run(p=0.1)
    assert pairing_errors(baseline, treatment) == []
    treatment["config"]["arguments"]["batch_size"] = 4
    assert "mismatched arguments" in pairing_errors(baseline, treatment)
    report = aggregate([baseline, _aggregate_run(p=0.1, complete=False)], [42, 43, 44])
    for group in report["groups"]:
        assert not group["complete_expected_seed_comparison"]
        if group["condition"]["p"] == 0.1:
            assert group["validation_loss"]["n"] == 0
            assert group["validation_loss"]["mean"] is None
        else:
            assert group["validation_loss"]["sample_std"] is None
    assert report["paired_comparisons"][0]["validation_loss_improvement"] is None


def test_aggregation_requires_dataset_provenance_and_reports_three_seeds():
    from scripts.aggregate_sparsedrop import aggregate, pairing_errors
    baseline, treatment = _aggregate_run(), _aggregate_run(p=0.1)
    baseline["config"]["data_manifest"] = {"provenance_status": "unavailable"}
    treatment["config"]["data_manifest"] = {"provenance_status": "unavailable"}
    assert "missing dataset/tokenizer manifest" in pairing_errors(baseline, treatment)
    report = aggregate([_aggregate_run(seed=seed) for seed in (42, 43, 44)], [42, 43, 44])
    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["complete_expected_seed_comparison"]
    assert group["validation_loss"] == {"n": 3, "mean": 4.0, "sample_std": 0.0}


def test_wall_time_segmentation_does_not_break_pairing():
    from scripts.aggregate_sparsedrop import pairing_errors
    baseline, treatment = _aggregate_run(), _aggregate_run(p=0.1)
    baseline['config']['arguments']['max_seconds'] = 100
    treatment['config']['arguments']['max_seconds'] = 200
    assert pairing_errors(baseline, treatment) == []


def test_duplicate_seed_markdown_identifies_contributors():
    from scripts.aggregate_sparsedrop import aggregate, render_markdown
    report = aggregate([_aggregate_run(seed=s) for s in [42, 42, 43, 44]], [42, 43, 44])
    group = report['groups'][0]
    assert group['contributing_seeds'] == [43, 44]
    assert group['validation_loss']['n'] == 2
    assert group['duplicate_seeds_excluded'] == [42]
    assert not group['complete_expected_seed_comparison']
    markdown = render_markdown(report)
    assert '[43, 44] / 2' in markdown
    assert '| [42] | False |' in markdown


def test_progress_plot_rejects_same_step_with_different_batch(tmp_path, monkeypatch):
    import json
    import sys
    pytest.importorskip('matplotlib')
    from scripts.plot_sparsedrop_progress import main
    for name, run in [
        ('h4_none_p0.00_constant_d0.00_s42', _aggregate_run(complete=False)),
        ('h4_both_p0.10_constant_d0.00_s42', _aggregate_run(p=0.1, complete=False)),
    ]:
        directory = tmp_path / name
        directory.mkdir()
        if run['condition']['p']:
            run['config']['arguments']['batch_size'] = 4
        (directory / 'config.json').write_text(json.dumps(run['config']))
        (directory / 'eval.jsonl').write_text(json.dumps(run['evaluation'])+'\n')
        (directory / 'train.jsonl').write_text(json.dumps(dict(event='train',step=5,loss=4.,grad_norm_preclip=1.,tokens_per_second=100.))+'\n')
    output = tmp_path / 'plot.png'
    monkeypatch.setattr(sys, 'argv', ['plot', str(tmp_path), '--output', str(output)])
    main()
    payload = json.loads(output.with_suffix('.json').read_text())
    assert payload['status'] == 'pairing_rejected'
    assert 'delta_loss_baseline_minus_dropout' not in payload


def _write_aggregate_files(directory, run, summary=None, train_rows=()):
    import json
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(run["config"]))
    (directory / "eval.jsonl").write_text(json.dumps(run["evaluation"]) + "\n")
    if summary is not None:
        (directory / "summary.json").write_text(json.dumps(summary))
    if train_rows:
        (directory / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows))


def _capped_aggregate_run(p=0.0):
    run = _aggregate_run(p=p)
    run["config"]["arguments"].update(max_steps=1_220_704, batch_size=32, grad_accum=1,
                                       seq_len=512, stop_after_steps=100_000)
    run["config"].update(token_budget=20_000_000_000, effective_target_steps=100_000,
                         effective_target_tokens=1_638_400_000)
    run["evaluation"].update(step=100_000, tokens_seen=1_638_400_000)
    summary = {"status": "complete", "step": 100_000, "tokens_seen": 1_638_400_000,
               "final_full_validation": run["evaluation"].copy()}
    return run, summary


def test_capped_runs_complete_and_pair_at_effective_budget(tmp_path):
    from scripts.aggregate_sparsedrop import aggregate, load_run
    loaded = []
    for p in (0.0, 0.1):
        run, summary = _capped_aggregate_run(p)
        directory = tmp_path / f"p{p}"
        _write_aggregate_files(directory, run, summary)
        loaded.append(load_run(directory))
    report = aggregate(loaded, [42])
    assert all(run["complete"] for run in loaded)
    assert all(group["validation_loss"]["n"] == 1 for group in report["groups"])
    assert report["paired_comparisons"][0]["status"] == "final_paired"
    assert report["paired_comparisons"][0]["validation_loss_improvement"] == pytest.approx(0.1)


def test_legacy_budget_and_default_cap_metadata_remain_compatible(tmp_path):
    import copy
    from scripts.aggregate_sparsedrop import comparison_config, load_run
    run = _aggregate_run()
    _write_aggregate_files(tmp_path / "legacy", run, {"status": "complete", "step": 10,
                           "tokens_seen": 160, "final_full_validation": run["evaluation"]})
    loaded = load_run(tmp_path / "legacy")
    assert loaded["complete"]
    assert loaded["effective_target"] == {"step": 10, "tokens_seen": 160}
    newer = copy.deepcopy(run["config"])
    newer["arguments"]["stop_after_steps"] = None
    newer.update(effective_target_steps=10, effective_target_tokens=160)
    assert comparison_config(run["config"]) == comparison_config(newer)


@pytest.mark.parametrize("step,status", [(99_999, "complete"), (147_609, "complete"),
                                         (147_609, "stopped_over_budget")])
def test_capped_final_rejects_inexact_budget_and_retains_actual_progress(tmp_path, step, status):
    from scripts.aggregate_sparsedrop import aggregate, load_run
    run, summary = _capped_aggregate_run(p=0.1)
    summary.update(status=status, step=step, tokens_seen=step * 16_384)
    run["evaluation"].update(step=step, tokens_seen=step * 16_384)
    _write_aggregate_files(tmp_path / "run", run, summary)
    loaded = load_run(tmp_path / "run")
    assert not loaded["complete"]
    assert loaded["training_progress"]["step"] == step
    assert loaded["evaluation"]["step"] == step
    assert loaded["budget_exceeded"] == (step > 100_000)
    assert aggregate([loaded], [42])["groups"][0]["validation_loss"]["n"] == 0


def test_over_budget_training_cannot_pair_using_stale_at_cap_evaluation(tmp_path):
    from scripts.aggregate_sparsedrop import load_run, pairing_errors
    baseline, baseline_summary = _capped_aggregate_run()
    treatment, _ = _capped_aggregate_run(p=0.1)
    _write_aggregate_files(tmp_path / "baseline", baseline, baseline_summary)
    _write_aggregate_files(tmp_path / "treatment", treatment, train_rows=[
        {"event": "checkpoint", "step": 147_609, "tokens_seen": 147_609 * 16_384}])
    baseline, treatment = load_run(tmp_path / "baseline"), load_run(tmp_path / "treatment")
    assert treatment["evaluation"]["step"] == 100_000
    assert treatment["training_progress"]["step"] == 147_609
    assert "treatment exceeds effective step/token budget" in pairing_errors(baseline, treatment)


def test_cap_metadata_requires_both_targets_and_preserves_lr_horizon_comparison(tmp_path):
    import copy
    from scripts.aggregate_sparsedrop import comparison_config, load_run
    run, summary = _capped_aggregate_run()
    different_horizon = copy.deepcopy(run["config"])
    different_horizon["arguments"]["max_steps"] = 100_000
    assert comparison_config(run["config"]) != comparison_config(different_horizon)
    del run["config"]["effective_target_tokens"]
    _write_aggregate_files(tmp_path / "run", run, summary)
    loaded = load_run(tmp_path / "run")
    assert not loaded["complete"]
    assert not loaded["budget_metadata_valid"]


def test_aggregation_ignores_archived_config_copies(tmp_path, monkeypatch):
    import json
    import sys
    from scripts.aggregate_sparsedrop import main
    run, summary = _capped_aggregate_run()
    live = tmp_path / "live"
    _write_aggregate_files(live, run, summary)
    for name in ("before_100k_cap_20261017", "recovery", "archive", "backups"):
        _write_aggregate_files(live / name, run, summary)
    monkeypatch.setattr(sys, "argv", ["aggregate", str(tmp_path)])
    main()
    report = json.loads((tmp_path / "analysis.json").read_text())
    assert report["number_of_runs"] == 1
    assert report["groups"][0]["duplicate_seeds_excluded"] == []
