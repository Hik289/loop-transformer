#!/usr/bin/env python3
"""Print a registered configuration and output directory for a Slurm array."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def select(study, index):
    folder = 'sparsedrop_400m20b' if study == '400m' else 'sparsedrop_long'
    rows = json.loads((ROOT / 'configs' / folder / 'manifest.json').read_text())
    if not 0 <= index < len(rows):
        raise ValueError(f'Index must be in [0, {len(rows)})')
    row = rows[index]
    if study == '400m':
        from scripts.validate_sparsedrop_400m_launch import validate
        return validate(row['host'], index)
    config = ROOT / row['config']
    cfg = json.loads(config.read_text())
    output = (ROOT / cfg['output_dir']).resolve()
    summary = output / 'summary.json'
    complete = False
    if summary.exists():
        s = json.loads(summary.read_text())
        complete = (s.get('status') == 'complete' and s.get('step') == 100000
                    and s.get('tokens_seen') == 1638400000)
        if not complete:
            raise ValueError('Existing summary does not match the 100k comparison budget')
    return config, output, complete

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=['80m', '400m'])
    parser.add_argument('index', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    config, output, complete = select(args.study, args.index)
    print(config)
    print(output)
    print('complete' if complete else 'train')
