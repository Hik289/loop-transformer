"""Checkpoint-bound execution changes retain training state and segmented cost."""
import argparse
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import train_sparsedrop as trainer
from tests.test_sparsedrop_training import tiny_arguments


def authorization(output, checkpoint, state, to):
    return {
        'schema_version': 1, 'output_dir': str(output.resolve()),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'checkpoint_step': state['step'], 'checkpoint_tokens_seen': state['tokens_seen'],
        'from': {k: state['arguments'][k] for k in trainer.EXECUTION_PROFILE_KEYS},
        'to': to, 'reason': 'Explicitly authorized test transition',
    }


@pytest.fixture
def scoped_transition(tmp_path):
    checkpoint = tmp_path / 'source.pt'
    checkpoint.write_bytes(b'checkpoint identity fixture')
    old = {'batch_size': 8, 'grad_accum': 4, 'gradient_checkpointing': True}
    new = {'batch_size': 32, 'grad_accum': 1, 'gradient_checkpointing': False}
    state = {'arguments': old, 'step': 100, 'tokens_seen': 1638400}
    record = authorization(tmp_path, checkpoint, state, new)
    mismatches = {k: (old[k], new[k]) for k in old}
    return tmp_path, checkpoint, state, new, record, mismatches


def test_execution_transition_requires_authorization(scoped_transition):
    output, source, state, new, record, mismatches = scoped_transition
    with pytest.raises(ValueError, match='no checkpoint-bound'):
        trainer.authorize_execution_transition(output, source, state, new, mismatches)


@pytest.mark.parametrize('field,value', [
    ('output_dir', '/different/run'), ('checkpoint_step', 101),
    ('checkpoint_tokens_seen', 1), ('checkpoint_sha256', '0' * 64),
    ('from', {'batch_size': 16, 'grad_accum': 2, 'gradient_checkpointing': True}),
    ('to', {'batch_size': 16, 'grad_accum': 2, 'gradient_checkpointing': False}),
])
def test_execution_transition_rejects_mis_scoped_authorization(scoped_transition, field, value):
    output, source, state, new, record, mismatches = scoped_transition
    record[field] = value
    (output / 'execution_transition.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        trainer.authorize_execution_transition(output, source, state, new, mismatches)


def test_execution_authorization_cannot_change_lr_or_effective_batch(scoped_transition):
    output, source, state, new, record, mismatches = scoped_transition
    (output / 'execution_transition.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Resume would change'):
        trainer.authorize_execution_transition(output, source, state, new,
                                              {**mismatches, 'lr': (0.0003, 0.001)})
    new['batch_size'] = 64
    record['to'] = new
    (output / 'execution_transition.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='effective optimizer batch'):
        trainer.authorize_execution_transition(output, source, state, new, mismatches)


def test_transition_preserves_state_and_accumulates_both_execution_segments(tmp_path, monkeypatch):
    args = tiny_arguments(tmp_path)
    output = tmp_path / 'run'
    assert trainer.main(args + ['--output-dir', str(output), '--stop-after-steps', '2']) == 0
    old = torch.load(output / 'last.pt', weights_only=False)
    # Exercise a real legacy checkpoint without the newly introduced fields.
    old.pop('approx_train_flops')
    old.pop('execution_history')
    source = output / 'source_checkpoint.pt'
    torch.save(old, source)
    original_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    new_profile = {'batch_size': 4, 'grad_accum': 1, 'gradient_checkpointing': False}
    (output / 'execution_transition.json').write_text(json.dumps(
        authorization(output, source, old, new_profile)))
    restored = set()
    original_model_load = trainer.LoopLM.load_state_dict
    original_optimizer_load = torch.optim.AdamW.load_state_dict
    original_sampler_load = trainer.ChunkSampler.load_state_dict
    original_restore_rng = trainer.restore_rng

    def model_load(self, values, *a, **kw):
        assert all(torch.equal(values[k], v) for k, v in old['model'].items())
        restored.add('model')
        return original_model_load(self, values, *a, **kw)

    def optimizer_load(self, values, *a, **kw):
        for key, states in old['optimizer']['state'].items():
            assert all(torch.equal(values['state'][key][k], v) for k, v in states.items())
        assert values['param_groups'] == old['optimizer']['param_groups']
        restored.add('optimizer')
        return original_optimizer_load(self, values, *a, **kw)

    def sampler_load(self, values):
        assert np.array_equal(values['order'], old['sampler']['order'])
        assert values['position'] == old['sampler']['position']
        assert values['rng'] == old['sampler']['rng']
        restored.add('sampler')
        return original_sampler_load(self, values)

    def restore_rng(values):
        if 'rng' not in restored:
            assert torch.equal(values['torch'], old['rng']['torch'])
            assert values['python'] == old['rng']['python']
            assert np.array_equal(values['numpy'][1], old['rng']['numpy'][1])
            restored.add('rng')
        return original_restore_rng(values)

    changed = args + ['--output-dir', str(output), '--batch-size', '4', '--grad-accum', '1',
                      '--no-gradient-checkpointing']
    with monkeypatch.context() as patch:
        patch.setattr(trainer.LoopLM, 'load_state_dict', model_load)
        patch.setattr(torch.optim.AdamW, 'load_state_dict', optimizer_load)
        patch.setattr(trainer.ChunkSampler, 'load_state_dict', sampler_load)
        patch.setattr(trainer, 'restore_rng', restore_rng)
        assert trainer.main(changed + ['--resume', str(source), '--stop-after-steps', '3']) == 0
    assert restored == {'model', 'optimizer', 'sampler', 'rng'}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_source_hash
    after = torch.load(output / 'last.pt', weights_only=False)
    old_factor = trainer.approximate_flops_per_token(argparse.Namespace(**old['arguments']))
    new_factor = trainer.approximate_flops_per_token(argparse.Namespace(**after['arguments']))
    expected = 2 * 32 * old_factor + 32 * new_factor
    assert after['step'] == 3 and after['tokens_seen'] == 96
    assert after['approx_train_flops'] == expected
    assert len(after['execution_history']) == 1
    event = after['execution_history'][0]
    assert event['step'] == 2 and event['tokens_seen'] == 64
    assert event['from']['gradient_checkpointing'] is True
    assert event['to'] == new_profile
    assert event['approx_train_flops'] == 64 * old_factor
    metadata = json.loads((output / 'config.json').read_text())
    assert metadata['restored_step'] == 2
    assert metadata['execution_history'] == after['execution_history']
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['approx_train_flops'] == expected
    assert summary['execution_history'] == after['execution_history']
    # A later ordinary resume carries prior cost/history without reauthorizing.
    assert trainer.main(changed + ['--resume', str(output / 'last.pt'), '--stop-after-steps', '4']) == 0
    continued = torch.load(output / 'last.pt', weights_only=False)
    assert continued['step'] == 4 and continued['tokens_seen'] == 128
    assert continued['approx_train_flops'] == expected + 32 * new_factor
    assert continued['execution_history'] == after['execution_history']
    events = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    assert len([row for row in events if row['event'] == 'execution_transition']) == 1
