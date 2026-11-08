#!/usr/bin/env python3
"""GPU correctness/performance smoke; synthetic tokens, never scientific results."""
import json
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = True
for checkpointing, batch_size in ((True, 8), (False, 8), (False, 32)):
    torch.manual_seed(42)
    cfg = LoopLMConfig.small()
    cfg.max_seq_len = 512
    cfg.max_recurrent_steps = 8
    cfg.residual_dropout_mode = 'both'
    cfg.residual_dropout_p = .1
    cfg.residual_dropout_schedule = 'increasing'
    cfg.residual_dropout_delta = .1
    model = LoopLM(cfg).cuda().train()
    model.gradient_checkpointing = checkpointing
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    tokens = torch.randint(0, cfg.vocab_size, (batch_size, 513), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        torch.cuda.synchronize()
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(tokens[:, :-1], final_only=True).logits[-1]
            loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        print(json.dumps({'synthetic_smoke_only': True, 'gpu': torch.cuda.get_device_name(), 'checkpointing': checkpointing,
                          'loops': 8, 'batch_size': batch_size, 'step': step+1, 'loss': loss.item(), 'grad_norm': grad.item(),
                          'seconds': time.monotonic()-start, 'peak_memory_gib': torch.cuda.max_memory_allocated()/2**30}), flush=True)
    del model, opt, tokens, logits, loss
    torch.cuda.empty_cache()
