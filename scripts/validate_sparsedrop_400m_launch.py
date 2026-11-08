#!/usr/bin/env python3
"""Check the frozen 18-run contract before a GPU job can start."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_KEYS = ('batch_size', 'grad_accum', 'gradient_checkpointing')


def execution_profile(index, root=None):
    """Require one explicit execution profile for every experiment.

    Batch grouping may differ after a measured optimization; the effective
    optimizer batch and the scientific token/LR budget remain fixed.
    """
    root = ROOT if root is None else Path(root)
    policy = json.loads((root / 'cluster/400m_execution_profiles.json').read_text())
    profiles = policy.get('profiles', {})
    if set(profiles) != {str(i) for i in range(18)}:
        raise ValueError('Execution policy must cover exactly all 18 experiment indices')
    for key, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile) != set(EXECUTION_KEYS):
            raise ValueError(f'Invalid execution profile fields for index {key}')
        if (type(profile['batch_size']) is not int or type(profile['grad_accum']) is not int
                or profile['batch_size'] <= 0 or profile['grad_accum'] <= 0
                or profile['batch_size'] * profile['grad_accum'] != 32
                or type(profile['gradient_checkpointing']) is not bool):
            raise ValueError(f'Invalid execution profile or effective batch for index {key}')
    if type(index) is not int or not 0 <= index < 18:
        raise ValueError('Experiment index must be an integer in [0, 18)')
    return dict(profiles[str(index)])


def validate(host, index):
    rows = json.loads((ROOT / 'configs/sparsedrop_400m20b/manifest.json').read_text())
    assert len(rows) == 18 and [r['index'] for r in rows] == list(range(18))
    assert len({r['name'] for r in rows}) == 18
    assert {(r['loops'], r['mode']) for r in rows} == {
        (h, mode) for h in (2, 4, 6, 8, 10, 12) for mode in ('ffn', 'attention', 'both')}
    assert all(sum(r['host'] == h for r in rows) == 9 for h in ('h100', 'h200'))
    row = rows[index]
    assert row['host'] == host
    cfg_path = ROOT / row['config']
    cfg = json.loads(cfg_path.read_text())
    expected = dict(vocab_size=49152, hidden_size=2048, num_layers=6, num_heads=32,
                    intermediate_size=5504, p=.1, seed=42, data_seed=100045,
                    schedule='constant', delta=0., token_budget=20_000_000_000,
                    max_steps=1_220_704, stop_after_steps=None, seq_len=512,
                    diagnostics=True, diagnostic_tokens=512, eval_chunks=128,
                    lr=3e-4, warmup_steps=2000, min_lr_ratio=.1, resume='auto')
    profile = execution_profile(index)
    expected.update(profile)
    assert all(cfg.get(k) == v for k, v in expected.items()), 'Scientific configuration mismatch'
    assert all(type(cfg[key]) is type(value) for key, value in profile.items()), 'Execution profile type mismatch'
    assert cfg['batch_size'] * cfg['grad_accum'] * cfg['seq_len'] == 16_384
    assert math.ceil(cfg['token_budget'] / 16_384) == cfg['max_steps']
    assert cfg['loops'] == row['loops'] and cfg['mode'] == row['mode']
    assert Path(cfg['output_dir']).resolve() == (ROOT / 'results_400m20b' / row['name']).resolve()
    assert Path(cfg['data_dir']).resolve() == (ROOT / 'data/fineweb_edu20b').resolve()
    hashes = json.loads((ROOT / 'cluster/400m_source_sha256.json').read_text())
    assert 'cluster/400m_execution_profiles.json' in hashes, 'Execution policy is absent from source hashes'
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    import numpy as np
    data = Path(cfg['data_dir'])
    meta = json.loads((data / 'metadata.json').read_text())
    assert meta['status'] == 'complete' and meta['dataset'] == 'HuggingFaceFW/fineweb-edu'
    assert meta['splits']['train']['tokens'] == 20_000_000_001
    for split, count in (('train', 20_000_000_001), ('validation', 1_048_577), ('test', 1_048_577)):
        arr = np.load(data / f'{split}.npy', mmap_mode='r')
        assert arr.dtype == np.uint16 and arr.shape == (count,), split
    output = Path(cfg['output_dir'])
    summary = output / 'summary.json'
    complete = False
    if summary.exists():
        s = json.loads(summary.read_text())
        complete = (s.get('status') == 'complete' and s.get('token_budget_completed') is True
                    and s.get('tokens_seen') == 20_000_000_000 and s.get('step') == 1_220_704
                    and s.get('completion_reason') == 'original_budget')
        assert complete, 'Unexpected partial or incompatible summary; inspect before resuming'
        assert math.isfinite(s['final_full_validation']['loss'])
    if not complete and (output / 'checkpoint_retention.json').exists():
        raise RuntimeError('Compacted weights cannot be used for optimizer resume')
    return cfg_path, output, complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', choices=['h100', 'h200'])
    parser.add_argument('index', type=int, choices=range(18))
    args = parser.parse_args()
    cfg, output, complete = validate(args.host, args.index)
    print(cfg)
    print(output)
    print('complete' if complete else 'train')
