#!/usr/bin/env python3
"""Generate the official tests of a Library Checker problem.

Usage: cases.py PROBLEM

Checks out yosupo06/library-checker-problems at a pinned commit (cached), runs its
generate.py, which builds in/, out/ and the checker and verifies every file against
the published hashes. Prints the problem directory.
"""
import fcntl
import os
import resource
import subprocess
import sys
from pathlib import Path

COMMIT = '1814c4e5205517e368bb57a8d1127eb961cfeaae'
REPO = 'https://github.com/yosupo06/library-checker-problems'
CACHE = Path(os.environ.get('LC_CACHE', Path.home() / '.cache' / 'lc-opt'))


def checkout() -> Path:
    root = CACHE / 'problems' / COMMIT[:12]
    if (root / 'generate.py').exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    git = ['git', '-C', str(root)]
    subprocess.run(git + ['init', '-q'], check=True)
    subprocess.run(git + ['fetch', '-q', '--depth', '1', REPO, COMMIT], check=True)
    subprocess.run(git + ['checkout', '-q', 'FETCH_HEAD'], check=True)
    return root


def find(root: Path, problem: str) -> Path:
    tomls = list(root.glob(f'*/{problem}/info.toml'))
    if len(tomls) != 1:
        sys.exit(f'unknown problem: {problem}')
    return tomls[0].parent


def unlimited_stack():
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def generate(problem: str) -> Path:
    """Return the problem directory with in/, out/ and the checker ready."""
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / 'cases.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = checkout()
        result = subprocess.run([sys.executable, 'generate.py', '-p', problem], cwd=root,
                                preexec_fn=unlimited_stack, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f'generate.py failed for {problem}:\n{result.stdout}{result.stderr}')
        return find(root, problem)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(generate(sys.argv[1]))
