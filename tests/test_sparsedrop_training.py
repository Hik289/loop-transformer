"""Tests for label alignment, data RNG isolation and exact dropout resume."""
import json
import os
import signal

import numpy as np
import pytest
import torch

from scripts import train_sparsedrop as trainer
from scripts.train_sparsedrop import ChunkSampler, TokenChunks, main, parse_args


def test_readme_hook_only_targets_production_results(tmp_path, monkeypatch):
    from src.training import sparsedrop_readme
    monkeypatch.setattr(trainer, "__file__", str(tmp_path / "scripts/train_sparsedrop.py"))
    calls = []
    monkeypatch.setattr(sparsedrop_readme, "update_readme_results",
                        lambda **kwargs: calls.append(kwargs) or {"counts": {}, "errors": []})
    log = tmp_path / "train.jsonl"
    trainer.update_result_readme(tmp_path / "test-output", log)
    assert not calls and not log.exists()
    trainer.update_result_readme(tmp_path / "results_fineweb20b/run", log)
    assert len(calls) == 1
    assert calls[0]["results_root"] == tmp_path / "results_fineweb20b"
    assert json.loads(log.read_text())["event"] == "readme_updated"


def test_readme_write_failure_does_not_invalidate_training(tmp_path, monkeypatch, capsys):
    from src.training import sparsedrop_readme
    monkeypatch.setattr(trainer, "__file__", str(tmp_path / "scripts/train_sparsedrop.py"))
    def failed_update(**kwargs):
        raise OSError("simulated document write failure")
    monkeypatch.setattr(sparsedrop_readme, "update_readme_results", failed_update)
    log = tmp_path / "train.jsonl"
    trainer.update_result_readme(tmp_path / "results_fineweb20b/run", log)
    assert json.loads(log.read_text())["event"] == "readme_update_failed"
    assert "simulated document write failure" in capsys.readouterr().err


def test_next_token_alignment_across_chunk_boundary(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(18, dtype=np.uint16))
    data = TokenChunks(path, 4, 32)
    inputs, targets = data.batch([0, 1, 3], torch.device("cpu"))
    assert inputs.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [12, 13, 14, 15]]
    assert targets.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [13, 14, 15, 16]]
    assert data.n_chunks == 4


def test_sampler_independent_rng_and_resume():
    first, second = ChunkSampler(7, 123), ChunkSampler(7, 123)
    a = first.take(19)
    torch.rand(1000)
    np.random.random(1000)
    assert np.array_equal(a, second.take(19))
    state = first.state_dict()
    expected = first.take(24)
    restored = ChunkSampler(7, 999)
    restored.load_state_dict(state)
    assert np.array_equal(expected, restored.take(24))


def test_training_resume_is_exact_with_dropout_and_checkpointing(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(9)
    np.save(data / "train.npy", rng.integers(0, 32, 257, dtype=np.uint16))
    np.save(data / "validation.npy", rng.integers(0, 32, 41, dtype=np.uint16))
    (data / "manifest.json").write_text(json.dumps({"tokenizer": "synthetic-v32", "split": "independent_rng_samples"}))
    common = ["--data-dir", str(data), "--device", "cpu", "--vocab-size", "32", "--hidden-size", "16", "--num-layers", "1", "--num-heads", "2", "--intermediate-size", "32", "--seq-len", "8", "--loops", "2", "--mode", "both", "--p", "0.2", "--seed", "13", "--max-steps", "4", "--batch-size", "2", "--grad-accum", "2", "--warmup-steps", "1", "--eval-every", "2", "--eval-chunks", "2", "--eval-batch-size", "1", "--checkpoint-every", "2", "--log-every", "1", "--diagnostic-tokens", "8"]
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    assert main(common + ["--output-dir", str(whole)]) == 0
    original_step = torch.optim.AdamW.step
    calls = {"count": 0}
    def interrupt_after_second_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 2:
            os.kill(os.getpid(), signal.SIGUSR1)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt_after_second_step)
        assert main(common + ["--output-dir", str(resumed)]) == 99
    assert main(common + ["--output-dir", str(resumed)]) == 0
    full_state = torch.load(whole / "last.pt", weights_only=False)
    resumed_state = torch.load(resumed / "last.pt", weights_only=False)
    assert full_state["step"] == resumed_state["step"] == 4
    assert full_state["tokens_seen"] == 4 * 2 * 2 * 8
    for name, value in full_state["model"].items():
        assert torch.equal(value, resumed_state["model"][name]), name
    assert torch.equal(full_state["rng"]["torch"], resumed_state["rng"]["torch"])
    assert np.array_equal(full_state["sampler"]["order"], resumed_state["sampler"]["order"])
    assert full_state["sampler"]["position"] == resumed_state["sampler"]["position"]
    assert full_state["best_loss"] == resumed_state["best_loss"]
    assert len(list(resumed.glob("*.pt"))) == 2
    summary = json.loads((resumed / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["final_full_validation"]["chunks"] == 5
    assert summary["final_full_validation"]["tokens"] == 40
    assert "diagnostics" in summary["final_full_validation"]
    config = json.loads((resumed / "config.json").read_text())
    assert config["data_manifest"]["tokenizer"] == "synthetic-v32"


def tiny_arguments(tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(17)
    np.save(data / "train.npy", rng.integers(0, 32, 257, dtype=np.uint16))
    np.save(data / "validation.npy", rng.integers(0, 32, 41, dtype=np.uint16))
    return ["--data-dir", str(data), "--device", "cpu", "--vocab-size", "32", "--hidden-size", "16", "--num-layers", "1", "--num-heads", "2", "--intermediate-size", "32", "--seq-len", "8", "--loops", "2", "--mode", "both", "--p", "0.2", "--seed", "13", "--max-steps", "6", "--batch-size", "2", "--grad-accum", "2", "--warmup-steps", "0", "--eval-every", "2", "--eval-chunks", "2", "--eval-batch-size", "1", "--checkpoint-every", "2", "--log-every", "1", "--no-diagnostics"]


def assert_same_training_state(expected, actual):
    assert expected["step"] == actual["step"]
    assert expected["tokens_seen"] == actual["tokens_seen"]
    assert expected["compute_tokens_seen"] == actual["compute_tokens_seen"]
    for name, value in expected["model"].items():
        assert torch.equal(value, actual["model"][name]), name
    for parameter, state in expected["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, actual["optimizer"]["state"][parameter][key]), key
    assert torch.equal(expected["rng"]["torch"], actual["rng"]["torch"])
    assert np.array_equal(expected["sampler"]["order"], actual["sampler"]["order"])
    assert expected["sampler"]["position"] == actual["sampler"]["position"]
    assert expected["sampler"]["epoch"] == actual["sampler"]["epoch"]
    assert expected["best_loss"] == actual["best_loss"]


@pytest.mark.parametrize("budget", [1, 8, 16, 17, 31, 32, 33, 51, 64, 65])
def test_exact_token_budget_masks_tail_and_avoids_unused_microbatches(tmp_path, monkeypatch, budget):
    args = tiny_arguments(tmp_path) + ["--token-budget", str(budget)]
    output = tmp_path / "run"
    calls = []
    original_ce = trainer.F.cross_entropy
    def inspect_targets(logits, targets, *a, **kw):
        if torch.is_grad_enabled():
            calls.append((int((targets != -100).sum()), targets.numel()))
            count = int((targets != -100).sum())
            assert (targets[:count] >= 0).all()
            assert (targets[count:] == -100).all()
        return original_ce(logits, targets, *a, **kw)
    monkeypatch.setattr(trainer.F, "cross_entropy", inspect_targets)
    assert main(args + ["--output-dir", str(output)]) == 0
    state = torch.load(output / "last.pt", weights_only=False)
    summary = json.loads((output / "summary.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert sum(valid for valid, _ in calls) == budget
    assert len(calls) == (budget + 15) // 16
    assert state["step"] == (budget + 31) // 32
    assert state["tokens_seen"] == summary["tokens_seen"] == config["token_budget"] == budget
    assert state["compute_tokens_seen"] == sum(size for _, size in calls) == ((budget + 7) // 8) * 8
    assert state["sampler"]["position"] == (budget + 7) // 8
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-5)
    assert summary["approx_train_flops"] == state["compute_tokens_seen"] * config["flops_per_token_estimate"]


def test_partial_accumulation_uses_token_weighted_gradient(tmp_path, monkeypatch):
    import copy
    args = tiny_arguments(tmp_path) + ["--token-budget", "17", "--mode", "none", "--p", "0", "--clip-grad", "1000000"]
    output = tmp_path / "run"
    initial, observed = {}, []
    original_save, original_step = trainer.save_checkpoint, torch.optim.AdamW.step
    def remember_initial(path, payload):
        if payload["step"] == 0:
            initial.update(copy.deepcopy(payload))
        return original_save(path, payload)
    def capture_gradient(self, *a, **kw):
        observed.extend(p.grad.detach().clone() for group in self.param_groups for p in group["params"])
        return original_step(self, *a, **kw)
    monkeypatch.setattr(trainer, "save_checkpoint", remember_initial)
    monkeypatch.setattr(torch.optim.AdamW, "step", capture_gradient)
    assert main(args + ["--output-dir", str(output)]) == 0
    reference = trainer.LoopLM(trainer.LoopLMConfig(**initial["model_config"]))
    reference.load_state_dict(initial["model"])
    for parameter in reference.exit_gate.parameters():
        parameter.requires_grad_(False)
    data = TokenChunks(tmp_path / "data" / "train.npy", 8, 32)
    x, y = data.batch(initial["sampler"]["order"][:3], torch.device("cpu"))
    y.reshape(-1)[17:] = -100
    logits = reference(x, num_steps=2, final_only=True).logits[-1]
    trainer.F.cross_entropy(logits.reshape(-1, 32), y.reshape(-1), reduction="sum").div(17).backward()
    parameters = [p for p in reference.parameters() if p.requires_grad]
    ordered = [p for p in parameters if p.ndim >= 2] + [p for p in parameters if p.ndim < 2]
    for gradient, parameter in zip(observed, ordered):
        torch.testing.assert_close(gradient, parameter.grad, atol=1e-7, rtol=2e-5)


def test_wall_time_segment_resume_matches_uninterrupted_partial_budget(tmp_path):
    args = tiny_arguments(tmp_path) + ["--token-budget", "65"]
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    assert main(args + ["--output-dir", str(whole)]) == 0
    assert main(args + ["--output-dir", str(resumed), "--max-seconds", "0.000000000001"]) == 99
    state = torch.load(resumed / "last.pt", weights_only=False)
    assert state["step"] == 1 and state["tokens_seen"] == 32
    assert not (resumed / "summary.json").exists()
    events = [json.loads(line) for line in (resumed / "train.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "interrupted" and events[-1]["signal"] == "max_seconds"
    assert main(args + ["--output-dir", str(resumed), "--max-seconds", "100000"]) == 0
    assert_same_training_state(torch.load(whole / "last.pt", weights_only=False), torch.load(resumed / "last.pt", weights_only=False))


def test_failure_before_first_update_restarts_from_initial_checkpoint(tmp_path, monkeypatch):
    args = tiny_arguments(tmp_path) + ["--token-budget", "33"]
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    assert main(args + ["--output-dir", str(whole)]) == 0
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise RuntimeError("transient first-step failure")
        patch.setattr(torch.optim.AdamW, "step", fail)
        with pytest.raises(RuntimeError, match="transient"):
            main(args + ["--output-dir", str(resumed)])
    initial = torch.load(resumed / "last.pt", weights_only=False)
    assert initial["step"] == initial["tokens_seen"] == initial["sampler"]["position"] == 0
    assert main(args + ["--output-dir", str(resumed)]) == 0
    assert_same_training_state(torch.load(whole / "last.pt", weights_only=False), torch.load(resumed / "last.pt", weights_only=False))


def test_rollback_archives_future_logs_and_stale_summary(tmp_path, monkeypatch):
    args = tiny_arguments(tmp_path)
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    assert main(args + ["--output-dir", str(whole)]) == 0
    original_save = trainer.save_checkpoint
    def fail_save(path, payload):
        if payload["step"] == 4:
            raise OSError("disk failure before atomic replace")
        return original_save(path, payload)
    with monkeypatch.context() as patch:
        patch.setattr(trainer, "save_checkpoint", fail_save)
        with pytest.raises(OSError, match="disk failure"):
            main(args + ["--output-dir", str(resumed)])
    (resumed / "summary.json").write_text(json.dumps({"status": "complete", "step": 4}))
    with monkeypatch.context() as patch:
        def stop_after_reconciliation(*args, **kwargs):
            raise RuntimeError("inspect recovered logs")
        patch.setattr(torch.optim.AdamW, "step", stop_after_reconciliation)
        with pytest.raises(RuntimeError, match="inspect recovered"):
            main(args + ["--output-dir", str(resumed)])
    assert not (resumed / "summary.json").exists()
    for name in ("train.jsonl", "eval.jsonl"):
        rows = [json.loads(line) for line in (resumed / name).read_text().splitlines()]
        assert all(row["step"] <= 2 for row in rows)
    archives = list((resumed / "recovery").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "summary.json").exists()
    assert [json.loads(line)["step"] for line in (archives[0] / "eval.jsonl").read_text().splitlines()] == [4]
    assert main(args + ["--output-dir", str(resumed)]) == 0
    rows = [json.loads(line) for line in (resumed / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows if row["event"] == "train"] == [1, 2, 3, 4, 5, 6]
    assert_same_training_state(torch.load(whole / "last.pt", weights_only=False), torch.load(resumed / "last.pt", weights_only=False))


def test_legacy_defaults_and_long_budget_horizon():
    base = ["--data-dir", "/unused", "--output-dir", "/unused"]
    assert parse_args(base).max_steps == 10000
    args = parse_args(base + ["--token-budget", "20000000000", "--batch-size", "32", "--grad-accum", "1", "--seq-len", "512"])
    assert args.max_steps == 1220704
    capped = parse_args(base + ["--token-budget", "20000000000", "--batch-size", "32", "--grad-accum", "1", "--seq-len", "512", "--stop-after-steps", "100000"])
    assert capped.max_steps == 1220704
    assert capped.token_budget == 20000000000
    assert capped.stop_after_steps == 100000


def test_stop_cap_finalizes_off_schedule_and_preserves_original_lr(tmp_path):
    args = tiny_arguments(tmp_path) + ["--token-budget", "321", "--stop-after-steps", "3", "--log-every", "10"]
    output = tmp_path / "run"
    assert main(args + ["--output-dir", str(output)]) == 0
    checkpoint = torch.load(output / "last.pt", weights_only=False)
    summary = json.loads((output / "summary.json").read_text())
    config = json.loads((output / "config.json").read_text())
    evaluations = [json.loads(line) for line in (output / "eval.jsonl").read_text().splitlines()]
    assert [(row["event"], row["step"]) for row in evaluations] == [("validation_fixed", 2), ("validation_fixed", 3), ("validation_full_final", 3)]
    assert checkpoint["step"] == summary["step"] == 3
    assert checkpoint["tokens_seen"] == summary["tokens_seen"] == 96
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == pytest.approx(trainer.lr_for_step(3, 11, 0, 3e-4, 0.1))
    assert summary["final_full_validation"]["chunks"] == 5
    for record in (summary, config):
        assert record["status"] == "complete"
        assert record["completion_reason"] == "stop_after_steps"
        assert record["stop_after_steps"] == record["effective_target_steps"] == record["actual_steps"] == 3
        assert record["effective_target_tokens"] == record["actual_tokens_seen"] == 96
        assert record["original_max_steps"] == record["lr_schedule_max_steps"] == 11
        assert record["original_token_budget"] == 321
        assert record["token_budget_completed"] is False
        assert record["overshoot_steps"] == record["overshoot_tokens"] == 0
    assert config["arguments"]["max_steps"] == 11
    assert config["token_budget"] == config["arguments"]["token_budget"] == 321
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows if row["event"] == "train"] == [1, 3]


def test_stop_cap_resumes_legacy_checkpoint_without_stop_field(tmp_path):
    args = tiny_arguments(tmp_path) + ["--token-budget", "321"]
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    assert main(args + ["--output-dir", str(whole), "--stop-after-steps", "3"]) == 0
    assert main(args + ["--output-dir", str(resumed), "--max-seconds", "0.000000000001"]) == 99
    legacy = torch.load(resumed / "last.pt", weights_only=False)
    assert legacy["step"] == 1
    del legacy["arguments"]["stop_after_steps"]
    torch.save(legacy, resumed / "last.pt")
    assert main(args + ["--output-dir", str(resumed), "--stop-after-steps", "3"]) == 0
    assert_same_training_state(torch.load(whole / "last.pt", weights_only=False), torch.load(resumed / "last.pt", weights_only=False))
    with pytest.raises(ValueError, match="Resume would change the experiment"):
        main(args + ["--output-dir", str(resumed), "--stop-after-steps", "4", "--lr", "0.001"])


@pytest.mark.parametrize("cap", [2, 3])
def test_checkpoint_at_or_beyond_stop_cap_only_runs_final_evaluation(tmp_path, monkeypatch, cap):
    args = tiny_arguments(tmp_path) + ["--token-budget", "321"]
    output = tmp_path / "run"
    original_step = torch.optim.AdamW.step
    updates = 0
    def interrupt_third_step(self, *positional, **keywords):
        nonlocal updates
        result = original_step(self, *positional, **keywords)
        updates += 1
        if updates == 3:
            os.kill(os.getpid(), signal.SIGUSR1)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt_third_step)
        assert main(args + ["--output-dir", str(output)]) == 99
    before = torch.load(output / "last.pt", weights_only=False)
    assert before["step"] == 3
    before["arguments"].pop("stop_after_steps")
    torch.save(before, output / "last.pt")
    original_evaluate = trainer.evaluate
    evaluated_chunks = []
    def record_evaluation(model, data, indices, *positional, **keywords):
        evaluated_chunks.append(len(indices))
        return original_evaluate(model, data, indices, *positional, **keywords)
    def forbid_update(*positional, **keywords):
        raise AssertionError("A checkpoint at or beyond the cap must not train")
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", forbid_update)
        patch.setattr(trainer, "evaluate", record_evaluation)
        assert main(args + ["--output-dir", str(output), "--stop-after-steps", str(cap)]) == 0
    assert evaluated_chunks == [2, 5]
    after = torch.load(output / "last.pt", weights_only=False)
    assert after["step"] == before["step"] == 3
    assert after["tokens_seen"] == before["tokens_seen"] == 96
    assert after["trained_seconds"] == before["trained_seconds"]
    # Final fixed validation may improve best_loss, but all training state is
    # preserved exactly, including optimizer moments and both random streams.
    assert after["best_loss"] <= before["best_loss"]
    before["best_loss"] = after["best_loss"]
    assert_same_training_state(before, after)
    for name in ("summary.json", "config.json"):
        result = json.loads((output / name).read_text())
        assert result["status"] == ("stopped_over_budget" if cap < 3 else "complete")
        assert result["effective_target_steps"] == cap
        assert result["effective_target_tokens"] == cap * 32
        assert result["actual_steps"] == 3 and result["actual_tokens_seen"] == 96
        assert result["overshoot_steps"] == 3 - cap
        assert result["overshoot_tokens"] == (3 - cap) * 32
        assert result["original_token_budget"] == 321
        assert result["token_budget_completed"] is False
    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in train_rows if row["event"] == "train"] == [1, 2, 3]


def test_resume_preserves_historical_peak_memory(tmp_path):
    args = tiny_arguments(tmp_path) + ["--token-budget", "33"]
    output = tmp_path / "run"
    assert main(args + ["--output-dir", str(output), "--max-seconds", "0.000000000001"]) == 99
    checkpoint = torch.load(output / "last.pt", weights_only=False)
    # A previous GPU segment's recorded peak must survive even when the next
    # invocation observes a lower peak. No CUDA device is needed for persistence.
    checkpoint["peak_gpu_memory_bytes"] = 123456789
    torch.save(checkpoint, output / "last.pt")
    assert main(args + ["--output-dir", str(output)]) == 0
    final = torch.load(output / "last.pt", weights_only=False)
    summary = json.loads((output / "summary.json").read_text())
    assert final["peak_gpu_memory_bytes"] == summary["peak_gpu_memory_bytes"] == 123456789


def test_recovery_archives_truncated_line_before_append(tmp_path):
    valid = json.dumps({"event": "train", "step": 2}) + "\n"
    # Even a syntactically complete row without its trailing newline would join
    # the next appended event into invalid JSON after an abrupt write failure.
    truncated = json.dumps({"event": "checkpoint", "step": 2})
    path = tmp_path / "train.jsonl"
    path.write_text(valid + truncated)
    recovery = trainer.reconcile_history(tmp_path, 2)
    assert path.read_text() == valid
    assert recovery["files"]["train.jsonl"] == {"retained_rows": 1, "archived_rows": 1}


@pytest.mark.parametrize("option,value", [("--token-budget", "0"), ("--token-budget", "-1"), ("--max-seconds", "0"), ("--max-seconds", "nan"), ("--max-seconds", "inf"), ("--stop-after-steps", "0"), ("--stop-after-steps", "-1")])
def test_invalid_budgets_are_rejected(option, value):
    with pytest.raises(SystemExit):
        parse_args(["--data-dir", "/unused", "--output-dir", "/unused", option, value])
