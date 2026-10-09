#!/usr/bin/env python3
"""Decide whether a pull request may merge, from the bench reports of its CI jobs.

Usage: verdict.py REPORT_DIR

Each report compares the main branch's main.cpp (first source) with the pull request's
(second source) on one CI machine. A problem's ratio is the geometric mean over its machines.
The pull request's ratio is the geometric mean over its problems, each weighted equally,
and must not exceed 1: the new version may not be slower at all. With one problem this is
that problem's ratio; with many (a lib/ change) noise averages out. Prints a markdown table;
exits 1 on a slowdown.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    log_ratios = defaultdict(list)  # problem -> [(cpu, log ratio)]
    for path in sorted(Path(sys.argv[1]).rglob('*.json')):
        report = json.loads(path.read_text())
        new = list(report['results'].values())[1]
        log_ratios[report['problem']].append((report['environment']['cpu'], math.log(new['ratio'])))

    print('| problem | new/old (geomean) | per machine |\n|---|---|---|')
    problem_logs = []
    for problem, rows in sorted(log_ratios.items()):
        problem_logs.append(sum(x for _, x in rows) / len(rows))
        machines = ', '.join(f'{cpu}: {math.exp(x):.4f}' for cpu, x in rows)
        print(f'| {problem} | {math.exp(problem_logs[-1]):.4f} | {machines} |')
    if not problem_logs:
        print('| (no timed problems) | | |')
        return 0
    overall = math.exp(sum(problem_logs) / len(problem_logs))
    slower = overall > 1
    print(f'| **all {len(problem_logs)}** | **{overall:.4f}**{" SLOWER" if slower else ""} | |')
    return 1 if slower else 0


if __name__ == '__main__':
    sys.exit(main())
