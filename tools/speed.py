#!/usr/bin/env python3
"""Speed dashboard: SPEED.md (one row per problem) and bench/<problem>.md (one row per test).

  speed.py bench [PROBLEM ...] [--rounds N]
      On a VM (Linux, Docker). Time each problem's main.cpp and the I/O floor (tools/floor.c) on
      every official test, N rounds in rotated order, with tools/judge.py's compiler and runner.
      Takes /tmp/bench.lock per problem, so other timing runs can go between problems.
      Writes bench/data/<host>/<problem>.json. Run it from its own checkout, not a shared one.
  speed.py render
      Anywhere. Read bench/data, fetch judged times from the Library Checker API, and write
      SPEED.md and bench/<problem>.md.

Off the judge's CPU (a machine with AVX-512), main.cpp is built with -march=x86-64-v3 plus the
extensions the judge has, as AGENTS.md asks. Other users appear by time only, never by name.
"""
from __future__ import annotations  # macOS ships Python 3.9

import argparse
import datetime
import fcntl
import hashlib
import json
import socket
import statistics
import sys
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'bench' / 'data'
API = 'https://v3.api.judge.yosupo.jp'
USER = 'Aiyiyi'
LOCK = '/tmp/bench.lock'
PORTABLE = '-march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes'
MACHINES = {'lc-amd': 'AMD', 'lc-intel': 'Intel'}  # host -> column; others use the host name
MAIN, FLOOR = 'main', 'floor'
HOST = socket.gethostname().split('.')[0]


def problem_dirs() -> dict[str, Path]:
    return {d.parent.name: d.parent for d in sorted(ROOT.glob('problems/*/*/main.cpp'))}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---- bench

def first_line(path: Path, limit: int = 40) -> str:
    with open(path, 'rb') as f:
        line = f.read(limit + 1).split(b'\n', 1)[0].decode(errors='replace').strip()
    return line if len(line) <= limit else line[:limit - 3] + '...'


def bench_problem(name: str, source: Path, rounds: int) -> dict:
    import judge  # Linux only (tomllib, Docker)
    problem = judge.Problem(name)
    with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
        work = Path(tmp)
        [built] = judge.build([str(source)], work)
        (work / FLOOR).mkdir()
        (work / 'floor.c').write_bytes((ROOT / 'tools' / 'floor.c').read_bytes())
        judge.docker(work, f'gcc -O2 -o {FLOOR}/main floor.c')
        (work / 'sizes').mkdir()
        for case in problem.cases:
            (work / 'sizes' / case).write_text(str(problem.output_size(case)))

        runs = []
        for r in range(rounds):
            order = [MAIN, FLOOR] if r % 2 == 0 else [FLOOR, MAIN]
            runs += [(prog, case) for prog in order for case in problem.cases]
        binary = {MAIN: built, FLOOR: FLOOR}
        with open(LOCK, 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            results = judge.run_all(work, problem, [(binary[prog], case) for prog, case in runs])

    cases = {case: {'input_bytes': problem.input(case).stat().st_size,
                    'output_bytes': problem.output_size(case),
                    'first_line': first_line(problem.input(case)),
                    'verdicts': [], MAIN: [], FLOOR: []} for case in problem.cases}
    for (prog, case), (verdict, ms, _) in zip(runs, results):
        cases[case][prog].append(ms)
        if prog == MAIN:
            cases[case]['verdicts'].append(verdict)
    env = judge.environment([str(source)])
    return {'problem': name, 'host': HOST, 'cpu': env['cpu'], 'compile': judge.COMPILE,
            'commit': env['commit'], 'main_sha': digest(source), 'rounds': rounds,
            'date': datetime.date.today().isoformat(), 'cases': cases}


def bench(args) -> int:
    import judge
    if 'avx512f' in open('/proc/cpuinfo').read().split():
        judge.COMPILE = judge.COMPILE.replace('-march=native', PORTABLE)
    dirs = problem_dirs()
    names = args.problems or list(dirs)
    out = DATA / HOST
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        report = bench_problem(name, dirs[name] / 'main.cpp', args.rounds)
        (out / f'{name}.json').write_text(json.dumps(report, indent=1) + '\n')
        print(f'{name}: {score(report):.1f} ms', flush=True)
    return 0


# ---- render

def score(report: dict) -> float:
    """Median over rounds of the slowest case, as the judge scores one run."""
    cases = report['cases'].values()
    return statistics.median(max(c[MAIN][r] for c in cases) for r in range(report['rounds']))


def failed(report: dict) -> bool:
    return any(v != 'OK' for c in report['cases'].values() for v in c['verdicts'])


def api(path: str, **query) -> dict:
    url = f'{API}/{path}?{urllib.parse.urlencode(query)}'
    request = urllib.request.Request(url, headers={'Referer': 'https://judge.yosupo.jp/'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def judged(problem: str) -> tuple[dict | None, float | None]:
    """Our fastest AC submission, and the fastest AC time by anyone else, on current tests."""
    ours = api('submissions', problem=problem, user=USER, status='AC', order='+time', limit=1000)
    mine = [s for s in ours['submissions'] if s['is_latest']]
    best = mine[0] if mine else None
    everyone = api('submissions', problem=problem, status='AC', order='+time', limit=ours['count'] + 100)
    others = [s['time'] for s in everyone['submissions'] if s['is_latest'] and s.get('user_name') != USER]
    return best, (others[0] * 1000 if others else None)


def load_reports() -> dict[str, dict[str, dict]]:
    """host -> problem -> report"""
    reports = {}
    for host in sorted(DATA.glob('*/')):
        reports[host.name] = {p.stem: json.loads(p.read_text()) for p in sorted(host.glob('*.json'))}
    return reports


def label(host: str) -> str:
    return MACHINES.get(host, host)


def size(n: int) -> str:
    for unit, scale in (('MB', 1 << 20), ('KB', 1 << 10)):
        if n >= scale:
            return f'{n / scale:.1f} {unit}'
    return f'{n} B'


def ms(value: float | None) -> str:
    return '—' if value is None else f'{value:.1f}' if value < 100 else f'{value:.0f}'


def machine_cell(report: dict | None, sha: str) -> str:
    if report is None:
        return '—'
    cell = 'fail' if failed(report) else ms(score(report))
    return cell if report['main_sha'] == sha else cell + ' *'


def render_speed(dirs: dict[str, Path], reports: dict, hosts: list[str]) -> str:
    rows = []
    with ThreadPoolExecutor(8) as pool:
        judgements = dict(zip(dirs, pool.map(judged, dirs)))
    for name, d in dirs.items():
        best, other = judgements[name]
        ours = best['time'] * 1000 if best else None
        ratio = ours / other if ours is not None and other else None
        sha = digest(d / 'main.cpp')
        link = f'[{ours:.0f}](https://judge.yosupo.jp/submission/{best["id"]})' if best else '—'
        cells = [f'[{name}](bench/{name}.md)', link, '—' if other is None else f'{other:.0f}',
                 '—' if ratio is None else f'{ratio:.2f}']
        cells += [machine_cell(reports[h].get(name), sha) for h in hosts]
        rows.append((-(ratio or 0), name, '| ' + ' | '.join(cells) + ' |'))
    machines = '; '.join(f'{label(h)}: {next(iter(reports[h].values()))["cpu"]}' for h in hosts if reports[h])
    lines = [
        '# Speed', '',
        f'Generated by `python3 tools/speed.py render` on {datetime.date.today().isoformat()}. '
        'Times in ms; a score is the slowest test.', '',
        '- Judged: our fastest AC submission (link). Best other: the fastest AC by anyone else on the '
        'current tests, time only.',
        '- Ratio: Judged / Best other; above 1 means behind. Rows go from the largest ratio down.',
        '- Machine columns: the current `main.cpp`, median score over rounds (`tools/speed.py bench`). '
        '`*`: measured on an older `main.cpp`; `fail`: some test was not OK.',
        f'- {machines}.' if machines else '- No machine measurements yet.', '',
        '| ' + ' | '.join(['Problem', 'Judged', 'Best other', 'Ratio'] + [label(h) for h in hosts]) + ' |',
        '|---|---:|---:|---:|' + '---:|' * len(hosts),
    ]
    return '\n'.join(lines + [row for _, _, row in sorted(rows)]) + '\n'


def render_problem(name: str, sha: str, reports: dict, hosts: list[str]) -> str:
    present = [h for h in hosts if name in reports[h]]
    lines = [f'# {name}', '',
             'Every official test, median over rounds, ms. Floor: start, map the input, write an output of '
             'the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, '
             'work and formatting.', '']
    for h in present:
        r = reports[h][name]
        stale = '' if r['main_sha'] == sha else ' Older `main.cpp` than the current one.'
        lines.append(f'- {label(h)}: {r["cpu"]}, `{r["compile"]}`, commit {r["commit"]}, {r["rounds"]} rounds, '
                     f'{r["date"]}. Score {ms(score(r))}.{stale}')
    if not present:
        return '\n'.join(lines + ['Not measured yet.']) + '\n'
    header = ['Test', 'Input', 'Output', 'First line']
    for h in present:
        header += [label(h), f'{label(h)} floor', f'{label(h)} compute']
    lines += ['', '| ' + ' | '.join(header) + ' |', '|---|---:|---:|---|' + '---:|' * 3 * len(present)]
    first = reports[present[0]][name]['cases']
    for case in sorted(first, key=lambda c: -statistics.median(first[c][MAIN])):
        info = first[case]
        cells = [case, size(info['input_bytes']), size(info['output_bytes']), f'`{info["first_line"]}`']
        for h in present:
            c = reports[h][name]['cases'].get(case)
            if c is None:
                cells += ['—'] * 3
                continue
            total, floor = statistics.median(c[MAIN]), statistics.median(c[FLOOR])
            bad = sorted({v for v in c['verdicts'] if v != 'OK'})
            cells += [ms(total) + (' ' + '/'.join(bad) if bad else ''), ms(floor), ms(total - floor)]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def render(args) -> int:
    dirs = problem_dirs()
    reports = load_reports()
    hosts = sorted(reports, key=lambda h: list(MACHINES).index(h) if h in MACHINES else len(MACHINES))
    (ROOT / 'SPEED.md').write_text(render_speed(dirs, reports, hosts))
    (ROOT / 'bench').mkdir(exist_ok=True)
    for name, d in dirs.items():
        (ROOT / 'bench' / f'{name}.md').write_text(render_problem(name, digest(d / 'main.cpp'), reports, hosts))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('bench')
    p.add_argument('problems', nargs='*')
    p.add_argument('--rounds', type=int, default=5)
    sub.add_parser('render')
    args = parser.parse_args()
    return bench(args) if args.command == 'bench' else render(args)


if __name__ == '__main__':
    sys.exit(main())
