#!/usr/bin/env python3
"""Check tiny CPU training and exact token accounting without downloading data."""
from pathlib import Path
import json
import tempfile
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_sparsedrop import main

if __name__ == '__main__':
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='loop-transformer-smoke-') as temporary:
        root = Path(temporary)
        data = root / 'data'
        data.mkdir()
        rng = np.random.default_rng(42)
        for split, count in [('train', 1025), ('validation', 65)]:
            np.save(data / f'{split}.npy', rng.integers(0, 32, count, dtype=np.uint16))
        args = ['--data-dir', str(data), '--output-dir', str(root / 'run'),
                '--device', 'cpu', '--vocab-size', '32', '--hidden-size', '16',
                '--num-layers', '1', '--num-heads', '2', '--intermediate-size', '32',
                '--loops', '2', '--mode', 'both', '--p', '0.1', '--seq-len', '8',
                '--batch-size', '2', '--grad-accum', '2', '--token-budget', '257',
                '--warmup-steps', '1', '--eval-every', '3', '--eval-chunks', '2',
                '--eval-batch-size', '1', '--checkpoint-every', '3', '--log-every', '3',
                '--diagnostic-tokens', '8', '--no-gradient-checkpointing']
        assert main(args) == 0
        summary = json.loads((root / 'run/summary.json').read_text())
        assert summary['tokens_seen'] == 257 and summary['token_budget_completed']
        assert (root / 'run/last.pt').is_file()
        print('PASS: CPU training, checkpoint save, diagnostics, and exactly 257 tokens.')
