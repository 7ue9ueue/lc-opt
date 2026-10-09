#!/usr/bin/env python3
"""Separate the judge's launch spikes from a submission's real times.

  spikes.py ID [ID ...]

About 5% of judged cases get a fixed +9 ms that does not depend on the program (see tools/spikes.md).
A case is a spike when it is 6-14 ms above the median of its peers: its class (same name without
the _NN suffix) if that has at least 3 cases, else the cases within 5% of its memory use.
Prints the spiked cases, the score without them, and the chance that a submission of the same code
is judged clean.
"""
from __future__ import annotations  # macOS ships Python 3.9

import json
import re
import statistics
import sys
import urllib.request
from collections import defaultdict

API = 'https://v3.api.judge.yosupo.jp'
SPIKE_MS = (6.0, 14.0)
SPIKE_SIZE_MS = 9.0
SPIKE_RATE = 0.052  # 210 of 4014 cases, 108 submissions, 2026-09-27 to 2026-10-09


def cases(submission: int) -> list[dict]:
    request = urllib.request.Request(f'{API}/submissions/{submission}', headers={'Referer': 'https://judge.yosupo.jp/'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)['case_results']


def peer_median(case: dict, results: list[dict], classes: dict[str, list[float]]) -> float | None:
    times = classes[re.sub(r'_\d+$', '', case['case'])]
    if len(times) < 3:
        times = [c['time'] * 1000 for c in results if abs(c['memory'] - case['memory']) <= 0.05 * case['memory']]
    return statistics.median(times) if len(times) >= 3 else None


def report(submission: int) -> None:
    results = cases(submission)
    classes = defaultdict(list)
    for case in results:
        classes[re.sub(r'_\d+$', '', case['case'])].append(case['time'] * 1000)
    spiked, base = [], {}
    for case in results:
        name, ms = case['case'], case['time'] * 1000
        median = peer_median(case, results, classes)
        if median is not None and SPIKE_MS[0] <= ms - median <= SPIKE_MS[1]:
            spiked.append(f'{name} {ms:.0f} (peers {median:.0f})')
            base[name] = median
        else:
            base[name] = ms
    clean = max(base.values())
    exposed = sum(ms + SPIKE_SIZE_MS > clean for ms in base.values())
    judged = max(c['time'] * 1000 for c in results)
    print(f'{submission}: judged {judged:.0f} ms, clean {clean:.0f} ms; '
          f'P(clean run) = {(1 - SPIKE_RATE) ** exposed:.2f} ({exposed} cases within {SPIKE_SIZE_MS:.0f} ms of the max)')
    for line in spiked:
        print(f'  spike: {line}')


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        report(int(arg))
    return 0


if __name__ == '__main__':
    sys.exit(main())
