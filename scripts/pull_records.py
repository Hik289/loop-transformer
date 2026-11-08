#!/usr/bin/env python3
"""Download scientific records from explicitly configured SSH hosts."""
import argparse
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', action='append', required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--study', choices=['80m', '400m', 'pilot'], required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not PurePosixPath(args.remote_root).is_absolute() or not re.fullmatch(r'[A-Za-z0-9_./-]+', args.remote_root):
        parser.error('--remote-root must be an absolute path using letters, numbers, underscores, dots, slashes, or hyphens')
    if any(not re.fullmatch(r'[A-Za-z0-9_.@-]+', host) or host.startswith('-') for host in args.host):
        parser.error('Invalid SSH host')
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir or root / 'outputs' / 'downloaded_records' / args.study
    folder = {'80m': 'results_fineweb20b', '400m': 'results_400m20b', 'pilot': 'results'}[args.study]
    remote = args.remote_root.rstrip('/')
    for host in args.host:
        destination = output / host
        for source, target, filters in [
            (folder, 'results', ['--exclude=*.pt*', '--exclude=*.pth', '--exclude=*.ckpt', '--exclude=*.safetensors', '--exclude=*.lock']),
            ('logs', 'logs', ['--include=400m_*', '--exclude=*'] if args.study == '400m' else []),
        ]:
            (destination / target).mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-az', *filters,
                            '--rsync-path=' + 'cd ' + shlex.quote(remote) + ' && rsync',
                            f'{host}:{remote}/{source}/', str(destination / target) + '/'],
                           stdin=subprocess.DEVNULL, check=True)
        command = 'cd ' + shlex.quote(remote) + ' && squeue --me'
        queue = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, command],
                               stdin=subprocess.DEVNULL, text=True, capture_output=True, check=True)
        (destination / 'queue.txt').write_text(queue.stdout)
    print(f'Saved records under {output}')

if __name__ == '__main__':
    main()
