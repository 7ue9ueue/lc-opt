#!/usr/bin/env python3
"""Judge-like testing and timing of Library Checker solutions. Needs Linux and Docker.

  judge.py test PROBLEM SOURCE
      Compile with the judge's command, run every official test, check each output.
  judge.py bench PROBLEM SOURCE [SOURCE ...] [--rounds N] [--cases K] [--json FILE]
      Time the sources against each other on the K slowest tests. Each round runs every
      source once, in rotated order. A source's score in a round is its slowest case,
      as on the judge. The ratio column is the median over rounds of each source's time
      divided by the first source's time in the same round.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import cases

IMAGE = 'gcc:15.2.0@sha256:3ae15afe768b06d0c0fe088d822ba5f8045c26630bdacc8d8e7713cf5d8e7289'
COMPILE = 'g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp'
TOOLS = Path(__file__).resolve().parent


class Problem:
    def __init__(self, name: str):
        self.dir = cases.generate(name)
        info = tomllib.loads((self.dir / 'info.toml').read_text())
        self.time_limit = info['timelimit']
        self.checker = self.dir / Path(info.get('checker', 'checker.cpp')).stem
        self.cases = sorted(p.stem for p in (self.dir / 'in').glob('*.in'))

    def accepts(self, case: str, output: Path) -> bool:
        args = [self.checker, self.dir / 'in' / f'{case}.in', output, self.dir / 'out' / f'{case}.out']
        return subprocess.run(args, capture_output=True).returncode == 0


def docker(work: Path, script: str, problem: Problem | None = None) -> str:
    """Run a shell script in the judge's compiler image with judge-like limits."""
    cmd = ['docker', 'run', '--rm', '--network', 'none', '--memory', '1g', '--memory-swap', '1g',
           '--ulimit', 'stack=-1:-1', '--cpuset-cpus', str(os.cpu_count() - 1),
           '--user', f'{os.getuid()}:{os.getgid()}',
           '-v', f'{work}:/w', '-w', '/w']
    if problem:
        cmd += ['-v', f'{problem.dir / "in"}:/in:ro']
    result = subprocess.run(cmd + [IMAGE, 'sh', '-ec', script], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f'docker failed:\n{result.stdout}{result.stderr}')
    return result.stdout


def build(sources: list[str], work: Path) -> list[str]:
    """Compile each source as work/<name>/main; return the names."""
    names = [f'{i}-{Path(src).stem}' for i, src in enumerate(sources)]
    for name, src in zip(names, sources):
        (work / name).mkdir()
        shutil.copy(src, work / name / 'main.cpp')
    shutil.copy(TOOLS / 'runner.c', work)
    docker(work, 'gcc -O2 -o runner runner.c\n' + ''.join(f'(cd {n} && {COMPILE})\n' for n in names))
    return names


def run_all(work: Path, problem: Problem, runs: list[tuple[str, str]]) -> list[tuple[str, float, int]]:
    """Run (name, case) pairs in order inside one container; return (verdict, ms, rss_kb) each."""
    lines = [f'mkdir -p out/{n}' for n in {name for name, _ in runs}]
    lines += [f'./runner {problem.time_limit} /in/{case}.in out/{name}/{case}.out {name}/main'
              for name, case in runs]
    rows = docker(work, '\n'.join(lines), problem).split()
    return [(rows[i], float(rows[i + 1]), int(rows[i + 2])) for i in range(0, len(rows), 3)]


def environment(sources: list[str]) -> dict:
    cpu = next((line.split(':', 1)[1].strip() for line in open('/proc/cpuinfo')
                if line.startswith('model name')), platform.machine())
    commit = subprocess.run(['git', '-C', str(TOOLS), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    return {'cpu': cpu, 'kernel': platform.release(), 'image': IMAGE, 'compile': COMPILE,
            'commit': commit or 'unknown',
            'sources': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest()[:16] for s in sources}}


def test(args) -> int:
    problem = Problem(args.problem)
    with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
        work = Path(tmp)
        [name] = build([args.source], work)
        results = run_all(work, problem, [(name, case) for case in problem.cases])
        failures = 0
        for case, (verdict, ms, kb) in zip(problem.cases, results):
            if verdict == 'OK' and not problem.accepts(case, work / 'out' / name / f'{case}.out'):
                verdict = 'WA'
            failures += verdict != 'OK'
            print(f'{verdict:3} {ms:9.1f} ms {kb / 1024:7.1f} MiB  {case}')
    slowest = max(ms for _, ms, _ in results)
    print(f'{"PASS" if failures == 0 else "FAIL"}: {len(results) - failures}/{len(results)} cases, '
          f'slowest {slowest:.1f} ms, limit {problem.time_limit * 1000:.0f} ms')
    return 1 if failures else 0


def bench(args) -> int:
    problem = Problem(args.problem)
    with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
        work = Path(tmp)
        names = build(args.sources, work)

        # Warmup: every source on every case. Checks all outputs and finds the slowest cases.
        warmup = [(name, case) for name in names for case in problem.cases]
        slowest = {}
        for (name, case), (verdict, ms, _) in zip(warmup, run_all(work, problem, warmup)):
            if verdict != 'OK' or not problem.accepts(case, work / 'out' / name / f'{case}.out'):
                sys.exit(f'{name} fails {case} ({verdict}); run judge.py test first')
            slowest[case] = max(slowest.get(case, 0.0), ms)
        chosen = sorted(problem.cases, key=slowest.get)[-args.cases:]

        runs = []  # (round, name, case)
        for r in range(args.rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            runs += [(r, name, case) for name in order for case in chosen]
        results = run_all(work, problem, [(name, case) for _, name, case in runs])

    scores = {name: {} for name in names}  # name -> round -> slowest case ms
    for (r, name, _), (_, ms, _) in zip(runs, results):
        scores[name][r] = max(scores[name].get(r, 0.0), ms)
    report = {'problem': args.problem, 'cases': chosen, 'rounds': args.rounds,
              'environment': environment(args.sources), 'results': {}}
    base = scores[names[0]]
    print(f'{report["environment"]["cpu"]}, {args.rounds} rounds, cases: {" ".join(chosen)}')
    print(f'{"source":30} {"median":>9} {"min":>9} {"max":>9} {"ratio":>7}')
    for name, src in zip(names, args.sources):
        values = sorted(scores[name].values())
        median = statistics.median(values)
        # Paired ratio: compare within each round, then take the median. Robust to drift.
        ratio = statistics.median(scores[name][r] / base[r] for r in base)
        report['results'][src] = {'median_ms': median, 'ratio': ratio, 'rounds_ms': values}
        print(f'{src[-30:]:30} {median:9.2f} {values[0]:9.2f} {values[-1]:9.2f} {ratio:7.4f}')
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('test')
    p.add_argument('problem')
    p.add_argument('source')
    p = sub.add_parser('bench')
    p.add_argument('problem')
    p.add_argument('sources', nargs='+')
    p.add_argument('--rounds', type=int, default=11)
    p.add_argument('--cases', type=int, default=3, help='number of slowest tests to time')
    p.add_argument('--json', help='write the full report here')
    args = parser.parse_args()
    return test(args) if args.command == 'test' else bench(args)


if __name__ == '__main__':
    sys.exit(main())
