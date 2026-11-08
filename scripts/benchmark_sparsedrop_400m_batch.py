#!/usr/bin/env python3
"""Isolated GPU throughput check; never reads/writes production checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--loops', type=int, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--checkpointing', type=int, choices=(0, 1), required=True)
    p.add_argument('--mode', choices=('ffn', 'attention', 'both'), default='both')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--steps', type=int, default=6)
    args = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from src.model.config import LoopLMConfig
    from src.model.looplm import LoopLM
    assert args.batch_size in (8, 16, 32)
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    # Leave room for the stopped production process and CUDA driver allocations.
    torch.cuda.set_per_process_memory_fraction(0.84, 0)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    np.random.seed(42)
    report = dict(vars(args), scientific_result=False, effective_sequences=32,
                  tokens_per_update=16384, grad_accum=32 // args.batch_size,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  cuda=torch.version.cuda,
                  total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    try:
        config = LoopLMConfig(vocab_size=49152, hidden_size=2048, num_layers=6,
            num_heads=32, intermediate_size=5504, max_seq_len=512,
            max_recurrent_steps=args.loops, dropout=0., beta_kl=0.,
            residual_dropout_mode=args.mode, residual_dropout_p=.1)
        model = LoopLM(config)
        model.gradient_checkpointing = bool(args.checkpointing)
        for parameter in model.exit_gate.parameters():
            parameter.requires_grad_(False)
        model.to(device='cuda', dtype=torch.float32).train()
        trainable = [q for q in model.parameters() if q.requires_grad]
        report['parameters'] = sum(q.numel() for q in model.parameters())
        assert report['parameters'] == 404279297
        opt = torch.optim.AdamW([
            {'params': [q for q in trainable if q.ndim >= 2], 'weight_decay': .1},
            {'params': [q for q in trainable if q.ndim < 2], 'weight_decay': 0.}],
            lr=3e-4, betas=(.9, .95), eps=1e-8, fused=True)
        data = np.load(ROOT / 'data/fineweb_edu20b/train.npy', mmap_mode='r')
        # Fixed real text, independent of each production sampler and run.
        ids = np.random.default_rng(100045).permutation(4096)
        durations, losses, norms = [], [], []
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.warmup + args.steps):
            torch.cuda.synchronize()
            began = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            step_loss = 0.
            for start in range(0, 32, args.batch_size):
                chosen = ids[step * 32 + start:step * 32 + start + args.batch_size]
                rows = np.stack([data[int(i)*512:int(i)*512+513] for i in chosen]).astype(np.int64)
                inputs = torch.from_numpy(rows[:, :-1].copy()).to('cuda')
                targets = torch.from_numpy(rows[:, 1:].copy()).to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(inputs, num_steps=args.loops, final_only=True)
                    logits = output.logits[-1]
                    loss = F.cross_entropy(logits.float().reshape(-1, 49152), targets.reshape(-1))
                assert bool(torch.isfinite(loss))
                weighted = loss / report['grad_accum']
                step_loss += float(loss.detach()) / report['grad_accum']
                weighted.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True))
            opt.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            if step >= args.warmup:
                durations.append(elapsed)
                losses.append(step_loss)
                norms.append(norm)
        report.update(status='ok', seconds_per_step=durations,
            tokens_per_second=16384 * len(durations) / sum(durations),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            losses=losses, grad_norms=norms)
        assert all(math.isfinite(x) for x in losses + norms + durations)
    except torch.OutOfMemoryError as exc:
        report.update(status='oom', error=str(exc),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
