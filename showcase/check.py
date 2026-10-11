#!/usr/bin/env python3
"""Check and time a showcase solution. Linux with Docker (a VM), from the repository root.

  showcase/check.py DIR [--stress N] [--rounds R] [--only samples|stress|time|windows]

DIR is showcase/<judge>/<problem> with
  main.cpp        the bundled solution (showcase/bundle.py)
  brute.cpp       a simple reference
  gen.py          'gen.py SEED small' prints a small random input, 'gen.py SEED max' a largest one
                  (SEED 0, 1, ... give the max cases worth timing; gen.py --max-cases prints how many)
  samples/*.in    with expected *.out
  check.py        optional: 'check.py IN OUT ANSWER' exits 0 when OUT is right (default: tokens equal)

Steps: build main.cpp with the judge's compiler and flags (JUDGES below), and on Codeforces also
with mingw-w64 run under Wine; samples; N small random cases against brute.cpp; then the max cases,
each run R times, median and max wall time (all runs under one hold of /tmp/bench.lock). Binaries and generated
inputs go to /tmp/showcase/<problem>.
"""
import argparse
import fcntl
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Compiler image and flags per judge, as close to the judge's as a Linux build gets.
JUDGES = {
    'atcoder': ('gcc:15.2.0', '-std=gnu++23 -O2 -march=native'),  # AtCoder C++23 (GCC 15.2.0)
    'codeforces': ('gcc:14', '-std=c++23 -O2'),  # GNU G++23 14.2 (64 bit, msys2); Windows: see mingw
    'luogu': ('gcc:13', '-std=c++20 -O2'),
    'loj': ('gcc:13', '-std=c++20 -O2'),
    'qoj': ('gcc:13', '-std=c++20 -O2'),
    'codechef': ('gcc:13', '-std=c++20 -O2'),
}
REFERENCE = ('gcc:15.2.0', '-std=c++23 -O2')
MINGW = 'x86_64-w64-mingw32-g++-posix'
WINE = '/usr/lib/wine/wine64'


def run(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


def build(image: str, flags: str, src: Path, out: Path) -> None:
    rel_src, rel_out = src.relative_to(ROOT), out
    cmd = (f'docker run --rm -v {ROOT}:/w -v {out.parent}:{out.parent} -w /w {image} '
           f'g++ {flags} -I. {rel_src} -o {rel_out}')
    r = run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'build failed: {src}\n{r.stderr[-4000:]}')


def same(problem: Path, inp: Path, out: str, expected: str) -> bool:
    checker = problem / 'check.py'
    if not checker.exists():
        return out.split() == expected.split()
    work = Path('/tmp/showcase/check')
    work.mkdir(parents=True, exist_ok=True)
    (work / 'out').write_text(out)
    (work / 'ans').write_text(expected)
    return run([sys.executable, str(checker), str(inp), str(work / 'out'), str(work / 'ans')]).returncode == 0


def execute(binary: list[str], inp: Path, timeout: float = 60) -> tuple[str, float]:
    start = time.perf_counter()
    with inp.open('rb') as f:
        r = run(binary, stdin=f, capture_output=True, timeout=timeout)
    elapsed = time.perf_counter() - start
    if r.returncode:
        sys.exit(f'{binary[-1]} failed on {inp}: exit {r.returncode}\n{r.stderr.decode()[-2000:]}')
    return r.stdout.decode(), elapsed


def generate(problem: Path, seed: int, kind: str, path: Path) -> Path:
    with path.open('w') as f:
        run([sys.executable, str(problem / 'gen.py'), str(seed), kind], stdout=f, check=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--stress', type=int, default=300)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--only')
    args = ap.parse_args()
    problem = (ROOT / args.dir).resolve()
    judge = problem.parent.name
    image, flags = JUDGES[judge]
    work = Path('/tmp/showcase') / problem.name
    work.mkdir(parents=True, exist_ok=True)
    steps = {args.only} if args.only else {'samples', 'stress', 'time', 'windows'}

    main_bin, brute_bin = work / 'main', work / 'brute'
    build(image, flags, problem / 'main.cpp', main_bin)
    print(f'built main.cpp with {image} {flags}')

    if 'samples' in steps:
        for inp in sorted((problem / 'samples').glob('[!.]*.in')):
            out, _ = execute([str(main_bin)], inp)
            ok = same(problem, inp, out, inp.with_suffix('.out').read_text())
            print(f'sample {inp.name}: {"OK" if ok else "WRONG"}')
            if not ok:
                return 1

    if 'windows' in steps and judge == 'codeforces' and shutil.which(MINGW):
        exe = work / 'main.exe'
        r = run([MINGW, '-O2', '-std=c++20', '-static', '-Wl,--stack=268435456', str(problem / 'main.cpp'),
                 '-o', str(exe)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'mingw build failed\n{r.stderr[-4000:]}')
        for inp in sorted((problem / 'samples').glob('[!.]*.in')):
            out, _ = execute([WINE, str(exe)], inp)
            ok = same(problem, inp, out, inp.with_suffix('.out').read_text())
            print(f'windows sample {inp.name}: {"OK" if ok else "WRONG"}')
            if not ok:
                return 1

    if 'stress' in steps and args.stress:
        build(*REFERENCE, problem / 'brute.cpp', brute_bin)
        for seed in range(args.stress):
            inp = generate(problem, seed, 'small', work / 'small.in')
            out, _ = execute([str(main_bin)], inp)
            expected, _ = execute([str(brute_bin)], inp)
            if not same(problem, inp, out, expected):
                shutil.copy(inp, work / 'failing.in')
                print(f'stress seed {seed}: WRONG (input in {work / "failing.in"})')
                return 1
        print(f'stress: {args.stress} cases OK')

    if 'time' in steps:
        cases = int(run([sys.executable, str(problem / 'gen.py'), '--max-cases'], capture_output=True,
                        text=True, check=True).stdout)
        inputs = [generate(problem, seed, 'max', work / f'max{seed}.in') for seed in range(cases)]
        with open('/tmp/bench.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # held for all runs: no waiting inside a measurement
            results = [[execute([str(main_bin)], inp)[1] * 1000 for _ in range(args.rounds)] for inp in inputs]
        for seed, (inp, times) in enumerate(zip(inputs, results)):
            print(f'max case {seed}: median {statistics.median(times):.0f} ms, max {max(times):.0f} ms '
                  f'({args.rounds} runs, {inp.stat().st_size} bytes in)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
