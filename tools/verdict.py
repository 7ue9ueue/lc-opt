#!/usr/bin/env python3
"""Decide whether a pull request may merge, from the bench reports of its CI jobs.

Usage: verdict.py REPORT_DIR

Each report compares the main branch's main.cpp (first source) with the pull request's
(second source) on one CI machine. Per problem, the geometric mean of the ratios must
not exceed TOLERANCE. Prints a markdown table; exits 1 on a slowdown.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

TOLERANCE = 1.02  # allowed new/old time ratio; below this, differences are noise


def main() -> int:
    ratios = defaultdict(list)  # problem -> [(cpu, ratio)]
    for path in sorted(Path(sys.argv[1]).rglob('*.json')):
        report = json.loads(path.read_text())
        new = list(report['results'].values())[1]
        ratios[report['problem']].append((report['environment']['cpu'], new['ratio']))

    ok = True
    print('| problem | new/old (geomean) | per machine |\n|---|---|---|')
    for problem, rows in sorted(ratios.items()):
        mean = math.exp(sum(math.log(r) for _, r in rows) / len(rows))
        ok &= mean <= TOLERANCE
        machines = ', '.join(f'{cpu}: {r:.3f}' for cpu, r in rows)
        print(f'| {problem} | {mean:.3f}{"" if mean <= TOLERANCE else " SLOWER"} | {machines} |')
    if not ratios:
        print('| (no timed problems) | | |')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
