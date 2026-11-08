#!/usr/bin/env python3
"""Record the current source/configuration hashes used by the launch check."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    files = []
    for folder in ['src', 'scripts', 'configs/sparsedrop_400m20b']:
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in ['.py', '.json', '.slurm', '.sh'])
    files.extend(p for p in (ROOT / 'cluster').iterdir()
                 if p.is_file() and p.suffix in ['.sh', '.slurm'])
    files.append(ROOT / 'cluster/400m_execution_profiles.json')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files)}
    (ROOT / 'cluster/400m_source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(f'Recorded {len(hashes)} source/configuration hashes.')
