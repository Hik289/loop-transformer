#!/usr/bin/env python3
"""Compact, read-only progress summary; no checkpoint tensors are loaded."""
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', nargs='?', type=Path, default=Path('results'))
args = parser.parse_args()
for directory in sorted(args.results.glob('*')):
    if not directory.is_dir():
        continue
    result = {'run': directory.name}
    config_path = directory / 'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        result['target_steps'] = config.get('effective_target_steps', config.get('arguments', {}).get('max_steps'))
        result['target_tokens'] = config.get('effective_target_tokens', config.get('token_budget'))
    for name, key in [('train.jsonl', 'train'), ('eval.jsonl', 'validation')]:
        source = directory / name
        if not source.exists():
            continue
        entries = []
        for line in source.read_text().splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a final line may be in the middle of being appended
        chosen = [x for x in entries if x.get('event') == 'train'] if key == 'train' else entries
        if chosen:
            last = chosen[-1]
            result[key] = {k: last[k] for k in ('time_utc', 'event', 'step', 'loss', 'perplexity', 'grad_norm_preclip', 'tokens_seen', 'tokens_per_second', 'train_seconds', 'peak_gpu_memory_bytes') if k in last}
            if key == 'train':
                result['first_train'] = {k: chosen[0][k] for k in ('time_utc', 'step', 'loss')}
                progress = [x for x in entries if x.get('event') in ('train', 'checkpoint', 'interrupted', 'resume', 'complete', 'stopped_over_budget') and isinstance(x.get('step'), int)]
                if progress:
                    result['actual_step'] = progress[-1]['step']
            diagnostics = last.get('diagnostics', {})
            if diagnostics.get('cross_loop'):
                result['cross_loop'] = {k: diagnostics['cross_loop'].get(k) for k in ('mean_state_cosine', 'mean_update_cosine', 'mean_linear_cka')}
    for name in ('last.pt', 'best.pt'):
        source = directory / name
        if source.exists():
            result[name] = {'bytes': source.stat().st_size, 'mtime': source.stat().st_mtime}
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result['status'] = summary.get('status', 'in_progress_or_not_started')
    result['complete'] = result['status'] == 'complete'
    if summary:
        result['actual_step'] = summary.get('step')
        result['actual_tokens'] = summary.get('tokens_seen')
        result['overshoot_steps'] = summary.get('overshoot_steps', 0)
    print(json.dumps(result), flush=True)
