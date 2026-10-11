#!/usr/bin/env python3
"""Time main.cpp against baseline.cpp on every max case. Linux with Docker, from the repository root.

  showcase/bench.py DIR... [--rounds R] [--timeout S]

Builds both with the judge's compiler and flags (check.py's JUDGES), then, holding
/tmp/bench.lock for the whole run, alternates the two R times per max case. Prints the median
wall time of each in ms, '>S s' for runs killed at the timeout, and whether the outputs agree.
"""
import argparse
import fcntl
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import JUDGES, ROOT, build, generate  # noqa: E402


def run(binary: Path, inp: Path, timeout: float) -> tuple[float | None, bytes]:
    start = time.perf_counter()
    try:
        with inp.open('rb') as f:
            r = subprocess.run([str(binary)], stdin=f, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, b''
    return time.perf_counter() - start, r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--timeout', type=float, default=30)
    args = ap.parse_args()
    for d in args.dirs:
        problem = (ROOT / d).resolve()
        image, flags = JUDGES[problem.parent.name]
        work = Path('/tmp/showcase') / problem.name
        work.mkdir(parents=True, exist_ok=True)
        build(image, flags, problem / 'main.cpp', work / 'main')
        build(image, flags, problem / 'baseline.cpp', work / 'baseline')
        cases = int(subprocess.run([sys.executable, str(problem / 'gen.py'), '--max-cases'], capture_output=True,
                                   text=True, check=True).stdout)
        for seed in range(cases):
            inp = generate(problem, seed, 'max', work / f'max{seed}.in')
            times = {'main': [], 'baseline': []}
            outputs = {}
            with open('/tmp/bench.lock', 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                for _ in range(args.rounds):
                    for name in times:
                        if times[name] and times[name][-1] is None:
                            continue  # timed out once: do not wait again
                        t, out = run(work / name, inp, args.timeout)
                        times[name].append(t)
                        outputs.setdefault(name, out)

            def show(ts):
                done = [t for t in ts if t is not None]
                return f'>{args.timeout:.0f} s' if not done else f'{statistics.median(done) * 1000:.0f} ms'

            agree = outputs.get('baseline') in (b'', outputs['main'])
            print(f'{d} max{seed}: main {show(times["main"])}, baseline {show(times["baseline"])}'
                  f'{"" if agree else ", OUTPUTS DIFFER"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
