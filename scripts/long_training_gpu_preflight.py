#!/usr/bin/env python3
"""Tiny exact-budget CUDA check within the allocated training GPU."""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_sparsedrop import main

with tempfile.TemporaryDirectory(prefix='long_gpu_preflight_', dir=os.environ['TMPDIR']) as temporary:
    root = Path(temporary)
    data = root / 'data'
    data.mkdir()
    np.save(data / 'train.npy', np.arange(129, dtype=np.uint16) % 32)
    np.save(data / 'validation.npy', np.arange(33, dtype=np.uint16) % 32)
    (data / 'metadata.json').write_text(json.dumps({'dataset': 'synthetic_cuda_preflight', 'tokenizer': 'integer-v32'}))
    output = root / 'result'
    arguments = ['--data-dir', str(data), '--output-dir', str(output), '--device', 'cuda',
        '--vocab-size', '32', '--hidden-size', '16', '--num-layers', '1', '--num-heads', '2',
        '--intermediate-size', '32', '--seq-len', '8', '--loops', '2', '--mode', 'both', '--p', '0.2',
        '--batch-size', '2', '--grad-accum', '1', '--token-budget', '17', '--warmup-steps', '1',
        '--eval-every', '2', '--checkpoint-every', '2', '--eval-chunks', '2', '--eval-batch-size', '1',
        '--diagnostic-tokens', '8', '--log-every', '1']
    status = main(arguments)
    summary = json.loads((output / 'summary.json').read_text())
    assert status == 0 and summary['status'] == 'complete'
    assert summary['tokens_seen'] == 17 and summary['step'] == 2
    assert summary['compute_tokens_seen'] == 24
    print('EXACT_TOKEN_CUDA_PREFLIGHT_PASSED', flush=True)
    capped_output = root / 'capped_result'
    status = main(arguments + ['--output-dir', str(capped_output), '--stop-after-steps', '1'])
    capped = json.loads((capped_output / 'summary.json').read_text())
    assert status == 0 and capped['status'] == 'complete'
    assert capped['step'] == 1 and capped['tokens_seen'] == 16
    assert capped['effective_target_steps'] == 1 and not capped['token_budget_completed']
    # The existing two-step result must be evaluated without one more update.
    status = main(arguments + ['--stop-after-steps', '1'])
    overshoot = json.loads((output / 'summary.json').read_text())
    assert status == 0 and overshoot['status'] == 'stopped_over_budget'
    assert overshoot['step'] == 2 and overshoot['tokens_seen'] == 17
    assert overshoot['overshoot_steps'] == 1
    print('STOP_CAP_CUDA_PREFLIGHT_PASSED', flush=True)
