#!/usr/bin/env python3
"""Time the I/O floor of the convolution problems: floor.cpp, built for each problem's input
layout, on the problem's largest official tests, with tools/judge.py's compiler and runner.
Needs Linux and Docker.

  floor.py [PROBLEM ...] [--io IO_HPP ...] [--fixed] [--rounds N] [--cases K]

Each --io file stands in for lib/io/io.hpp, so library versions can be compared in the same run
(default: this repository's). --fixed prints answers in fixed-width fields where every value is
below 10^9 (a judge-specific trick, ./convolution_mod/fields.hpp). A round runs every version
once on each case, in rotated order; its score is the slowest case. Prints the median over rounds
and the median paired ratio to the first version.
"""
import argparse
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import bundle  # noqa: E402
import judge  # noqa: E402

# problem: (LAYOUT, VALUE, SUMS, every value < 10^9); see floor.cpp
PROBLEMS = {
    'convolution_mod': (1, 'std::uint32_t', False, True),
    'convolution_mod_1000000007': (1, 'std::uint32_t', False, False),
    'convolution_mod_large': (1, 'std::uint32_t', False, True),
    'convolution_mod_2_64': (1, 'std::uint64_t', False, False),
    'convolution_F_2_64': (1, 'std::uint64_t', False, False),
    'min_plus_convolution_convex_convex': (1, 'std::uint32_t', True, False),
    'min_plus_convolution_convex_arbitrary': (1, 'std::uint32_t', True, False),
    'min_plus_convolution_concave_arbitrary': (1, 'std::uint32_t', True, False),
    'bitwise_and_convolution': (2, 'std::uint32_t', False, True),
    'bitwise_xor_convolution': (2, 'std::uint32_t', False, True),
    'mul_mod2n_convolution': (2, 'std::uint32_t', False, True),
    'gcd_convolution': (3, 'std::uint32_t', False, True),
    'lcm_convolution': (3, 'std::uint32_t', False, True),
    'mul_modp_convolution': (3, 'std::uint32_t', False, True),
    'multivariate_convolution': (4, 'std::uint32_t', False, True),
    'multivariate_convolution_cyclic': (5, 'std::uint32_t', False, True),
}


def inline(path: Path, io_hpp: Path, seen: set[Path]) -> list[str]:
    """path with its quoted includes inlined once each, io_hpp in place of lib/io/io.hpp."""
    lines = []
    for line in path.read_text().splitlines():
        match = bundle.INCLUDE.match(line)
        if match:
            header = io_hpp if match.group(1) == 'lib/io/io.hpp' else bundle.resolve(match.group(1), path)
            if header not in seen:
                seen.add(header)
                lines += inline(header, io_hpp, seen)
        elif line.strip() != '#pragma once':
            lines.append(line)
    return lines


def source(problem: str, io_hpp: Path, fixed: bool) -> str:
    layout, value, sums, small = PROBLEMS[problem]
    defines = [f'#define LAYOUT {layout}', f'#define VALUE {value}']
    defines += ['#define SUMS'] * sums + ['#define FIXED'] * (fixed and small)
    return '\n'.join(defines + inline(HERE / 'floor.cpp', io_hpp, set())) + '\n'


def floor(problem: str, versions: list[Path], fixed: bool, rounds: int, count: int) -> None:
    p = judge.Problem(problem)
    size = {c: p.input(c).stat().st_size + p.output_size(c) for c in p.cases}
    chosen = sorted(p.cases, key=size.get)[-count:]
    with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
        work = Path(tmp)
        paths = []
        for i, io_hpp in enumerate(versions):
            path = work / f'floor{i}.cpp'
            path.write_text(source(problem, io_hpp, fixed))
            paths.append(str(path))
        names = judge.build(paths, work)
        runs = []
        for r in range(rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            runs += [(r, name, case) for name in order for case in chosen]
        results = judge.run_all(work, p, [(name, case) for _, name, case in runs])
    scores = {name: {} for name in names}
    for (r, name, case), (verdict, ms, _) in zip(runs, results):
        if verdict != 'OK':
            sys.exit(f'{problem}: {name} {verdict} on {case}')
        scores[name][r] = max(scores[name].get(r, 0.0), ms)
    largest = chosen[-1]
    output = 'fixed-width' if fixed and PROBLEMS[problem][3] else 'write_array'
    print(f'{problem}: cases {" ".join(chosen)}; largest in {p.input(largest).stat().st_size / 1e6:.2f} MB, '
          f'out {p.output_size(largest) / 1e6:.2f} MB; {output}')
    base = scores[names[0]]
    for name, io_hpp in zip(names, versions):
        values = sorted(scores[name].values())
        ratio = statistics.median(scores[name][r] / base[r] for r in base)
        print(f'  {str(io_hpp)[-40:]:40} median {statistics.median(values):8.2f} ms  '
              f'min {values[0]:8.2f}  max {values[-1]:8.2f}  ratio {ratio:.4f}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('problems', nargs='*', default=list(PROBLEMS))
    parser.add_argument('--io', type=Path, action='append', help='io.hpp version (repeatable)')
    parser.add_argument('--fixed', action='store_true', help='fixed-width output where values allow')
    parser.add_argument('--rounds', type=int, default=11)
    parser.add_argument('--cases', type=int, default=3, help='number of largest tests to time')
    args = parser.parse_args()
    versions = [v.resolve() for v in args.io] if args.io else [ROOT / 'lib/io/io.hpp']
    for problem in args.problems:
        floor(problem, versions, args.fixed, args.rounds, args.cases)
    return 0


if __name__ == '__main__':
    sys.exit(main())
