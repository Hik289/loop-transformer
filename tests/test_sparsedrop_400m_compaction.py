"""Destructive retention must preserve exact weights and recover interruptions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts import compact_sparsedrop_400m_checkpoints as retention


@pytest.fixture
def completed(tmp_path):
    contract = retention._ModelContract(vocab=16, hidden=4, layers=1, heads=2,
                                       intermediate=8, parameters=249)
    shapes = retention._expected_shapes(contract)
    # 16*4 embedding + (4*4*4 + 3*4*8 + 4*4) + 4+1 gate + 4 norm = 249.
    args = {"vocab_size": 16, "hidden_size": 4, "num_layers": 1, "num_heads": 2,
            "intermediate_size": 8, "loops": 12, "mode": "both", "p": .1,
            "schedule": "constant", "delta": 0., "seed": 42,
            "token_budget": 20_000_000_000, "batch_size": 8, "grad_accum": 4,
            "seq_len": 512, "gradient_checkpointing": True, "max_steps": 1_220_704,
            "stop_after_steps": None, "resume": "auto", "output_dir": str(tmp_path)}
    model = {key: args[key] for key in ("vocab_size", "hidden_size", "num_layers",
                                       "num_heads", "intermediate_size")}
    model.update(max_recurrent_steps=12, max_seq_len=512, residual_dropout_mode="both",
                 residual_dropout_p=.1, residual_dropout_schedule="constant",
                 residual_dropout_delta=0., dropout=0., beta_kl=0.)
    config = {"parameters_total": 249, "arguments": args, "model": model,
              "status": "complete", "token_budget_completed": True,
              "initial_parameters_sha256": "initial-model-hash", "data_manifest": {"source": "fixture"}}
    summary = {"status": "complete", "tokens_seen": 20_000_000_000,
               "actual_tokens_seen": 20_000_000_000, "effective_target_tokens": 20_000_000_000,
               "original_token_budget": 20_000_000_000, "token_budget_completed": True,
               "completion_reason": "original_budget", "step": 1_220_704,
               "actual_steps": 1_220_704, "effective_target_steps": 1_220_704,
               "overshoot_steps": 0, "overshoot_tokens": 0,
               "last_checkpoint": str(tmp_path / "last.pt"),
               "best_checkpoint": str(tmp_path / "best.pt"),
               "final_full_validation": {"loss": 3.2, "diagnostics": {"retained": True}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    originals = {}
    for role, step in (("best", 1_220_000), ("last", 1_220_704)):
        values = {name: torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape)
                  + (0.125 if role == "best" else 0.25) for name, shape in shapes.items()}
        values["lm_head.weight"] = values["embed.weight"]
        checkpoint = {"format_version": 2, "model": values, "model_config": deepcopy(model),
                      "arguments": deepcopy(args), "step": step,
                      "tokens_seen": min(step * 16384, 20_000_000_000),
                      "compute_tokens_seen": min(step * 16384, 20_000_000_000),
                      "initial_parameters_sha256": config["initial_parameters_sha256"],
                      "trained_seconds": 123.5, "best_loss": 3.1,
                      "data_signatures": {"train": {"bytes": 40000000000}},
                      "optimizer": {"state": torch.arange(10000)},
                      "sampler": {"order": torch.arange(2000)}, "rng": {"torch": torch.arange(32)}}
        torch.save(checkpoint, tmp_path / f"{role}.pt")
        originals[role] = checkpoint
    return tmp_path, contract, config, summary, originals


def run(fixture):
    output, contract, *_ = fixture
    return retention._compact(output, contract=contract)


def test_complete_retention_preserves_every_weight_and_metadata_and_is_idempotent(completed):
    output, _, config, old_summary, originals = completed
    report = run(completed)
    assert report["status"] == "complete" and report["source_checkpoints_removed"]
    for role in ("best", "last"):
        assert not (output / f"{role}.pt").exists()
        weights = torch.load(output / f"{role}_model.pt", map_location="cpu", weights_only=False)
        assert weights["weights_only"] is True and weights["resumable"] is False
        assert not retention.EXCLUDED_STATE.intersection(weights)
        assert weights["metadata"] == config
        retention._verify_equal(originals[role], weights, torch)
    updated = json.loads((output / "summary.json").read_text())
    assert updated["final_full_validation"] == old_summary["final_full_validation"]
    assert updated["last_checkpoint"] == str(output / "last_model.pt")
    assert updated["checkpoint_retention"]["original_checkpoint_paths"]["best_checkpoint"] == old_summary["best_checkpoint"]
    before = {p.name: p.read_bytes() for p in output.glob("*_model.pt")}
    rerun = run(completed)
    assert rerun["checkpoints"] == report["checkpoints"]
    assert before == {p.name: p.read_bytes() for p in output.glob("*_model.pt")}


@pytest.mark.parametrize("field,value", [("status", "running"), ("tokens_seen", 1_638_400_000),
    ("token_budget_completed", False), ("completion_reason", "stop_after_steps"),
    ("overshoot_steps", 1), ("step", 100000)])
def test_incomplete_or_wrong_budget_is_rejected_without_touching_sources(completed, field, value):
    output, _, _, summary, _ = completed
    summary[field] = value
    (output / "summary.json").write_text(json.dumps(summary))
    before = {p.name: p.read_bytes() for p in output.glob("*.pt")}
    with pytest.raises(ValueError):
        run(completed)
    assert before == {p.name: p.read_bytes() for p in output.glob("*.pt")}


def test_public_entrypoint_cannot_accept_tiny_or_80m_models(completed):
    output, *_ = completed
    with pytest.raises(ValueError, match="parameter count"):
        retention.compact(output)
    assert (output / "best.pt").exists() and (output / "last.pt").exists()


def test_changed_tensor_in_existing_replacement_preserves_both_originals(completed):
    output, _, config, _, originals = completed
    weights = {key: deepcopy(value) for key, value in originals["last"].items()
               if key not in retention.EXCLUDED_STATE}
    weights.update(weights_only=True, resumable=False, metadata=config)
    weights["model"]["final_norm.weight"][0] += 1
    torch.save(weights, output / "last_model.pt")
    with pytest.raises(ValueError, match="changed model tensor"):
        run(completed)
    assert (output / "best.pt").exists() and (output / "last.pt").exists()
    assert not (output / retention.RETENTION_FILE).exists()


def test_disk_write_failure_keeps_both_source_checkpoints(completed, monkeypatch):
    output, *_ = completed
    save = retention._save_weights
    def fail_second(path, value, torch_module):
        if path.name == "last_model.pt":
            raise OSError("disk full")
        save(path, value, torch_module)
    monkeypatch.setattr(retention, "_save_weights", fail_second)
    with pytest.raises(OSError, match="disk full"):
        run(completed)
    assert (output / "best.pt").exists() and (output / "last.pt").exists()


def test_interruption_after_first_source_removal_recovers_from_durable_proof(completed, monkeypatch):
    output, *_ = completed
    unlink = Path.unlink
    def fail_last(path, *args, **kwargs):
        if path == output / "last.pt":
            raise OSError("simulated interruption")
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_last)
        with pytest.raises(OSError, match="interruption"):
            run(completed)
    assert not (output / "best.pt").exists() and (output / "last.pt").exists()
    assert json.loads((output / retention.RETENTION_FILE).read_text())["status"] == "verified"
    assert run(completed)["status"] == "complete"
    assert not (output / "last.pt").exists()


def test_missing_source_requires_prior_proof_and_corrupt_retained_file_is_rejected(completed):
    output, *_ = completed
    run(completed)
    path = output / "best_model.pt"
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="missing or changed"):
        run(completed)


def test_unverified_missing_source_is_rejected(completed):
    output, *_ = completed
    (output / "best.pt").unlink()
    with pytest.raises(ValueError, match="without a prior verified"):
        run(completed)
    assert (output / "last.pt").exists()


def test_checkpoint_model_configuration_mismatch_keeps_sources(completed):
    output, _, _, _, originals = completed
    originals["best"]["model_config"]["num_layers"] = 7
    torch.save(originals["best"], output / "best.pt")
    with pytest.raises(ValueError, match="model configuration differs"):
        run(completed)
    assert (output / "best.pt").exists() and (output / "last.pt").exists()


def transitioned_fixture(completed, *, transition_step=1_220_001):
    output, _, config, _, originals = completed
    old = {key: config['arguments'][key] for key in retention.EXECUTION_KEYS}
    new = {'batch_size': 32, 'grad_accum': 1, 'gradient_checkpointing': False}
    history = [{'step': transition_step, 'tokens_seen': transition_step * 16384,
                'from': old, 'to': new, 'checkpoint_sha256': 'a' * 64}]
    config['arguments'].update(new)
    config['execution_history'] = history
    originals['last']['arguments'].update(new)
    originals['last']['execution_history'] = deepcopy(history)
    (output / 'config.json').write_text(json.dumps(config))
    torch.save(originals['last'], output / 'last.pt')
    return old, new


def test_historical_best_retains_original_profile_after_approved_transition(completed):
    output, _, config, _, originals = completed
    old, new = transitioned_fixture(completed)
    report = run(completed)
    assert report['status'] == 'complete'
    best = torch.load(output / 'best_model.pt', weights_only=False)
    last = torch.load(output / 'last_model.pt', weights_only=False)
    assert {key: best['arguments'][key] for key in retention.EXECUTION_KEYS} == old
    assert {key: last['arguments'][key] for key in retention.EXECUTION_KEYS} == new
    assert best['metadata']['execution_history'] == config['execution_history']
    retention._verify_equal(originals['best'], best, torch)


@pytest.mark.parametrize('fault', ['late_best', 'no_history', 'wrong_from', 'nonexecution', 'old_last'])
def test_transition_never_licenses_unrelated_or_late_checkpoint_changes(completed, fault):
    output, _, config, _, originals = completed
    transitioned_fixture(completed, transition_step=1_219_999 if fault == 'late_best' else 1_220_001)
    if fault == 'no_history':
        config.pop('execution_history')
    elif fault == 'wrong_from':
        config['execution_history'][0]['from']['batch_size'] = 16
        config['execution_history'][0]['from']['grad_accum'] = 2
    elif fault == 'nonexecution':
        originals['best']['arguments']['seed'] = 43
        torch.save(originals['best'], output / 'best.pt')
    elif fault == 'old_last':
        originals['last']['arguments'].update(batch_size=8, grad_accum=4, gradient_checkpointing=True)
        torch.save(originals['last'], output / 'last.pt')
    (output / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError):
        run(completed)
    assert (output / 'best.pt').exists() and (output / 'last.pt').exists()
