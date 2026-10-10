#!/usr/bin/env python3
"""Separate the judge's launch spikes from a submission's real times.

  spikes.py ID [ID ...]

About 5% of judged cases get a fixed +9 ms that does not depend on the program (see tools/spikes.md).
Real cost repeats from run to run; a spike does not. A case is a spike when it is 6-14 ms above its
reference and the reference is under 100 ms. The reference is the median time of the same case in
the problem's other AC submissions by Aiyiyi with the same source; without those, in the ones with
another source that match this run on the case's peers (median gap at most 1 ms). Without either,
the case is not flagged: when in doubt, the tool reports the slower score.
Peers: the other cases of the class (same name without the _NN suffix) if there are at least 3,
else the cases within 5% of its memory use.
Prints the spiked cases, the score without them, and the chance that a submission of the same code
is judged clean.
"""
from __future__ import annotations  # macOS ships Python 3.9

import functools
import hashlib
import json
import re
import statistics
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

API = 'https://v3.api.judge.yosupo.jp'
USER = 'Aiyiyi'
SPIKE_MS = (6.0, 14.0)
NOISY_MS = 100.0  # from here on, runs of the same code differ by up to 6 ms and hide a spike
VERSION_GAP_MS = 1.0
SPIKE_SIZE_MS = 9.0
SPIKE_RATE = 0.052  # 210 of 4014 cases, 108 submissions, 2026-09-27 to 2026-10-09


@dataclass
class Run:
    id: int
    problem: str
    user: str
    source: str  # digest
    times: dict[str, float]  # ms, in judge order
    memory: dict[str, int]  # bytes


def get(path: str) -> dict:
    request = urllib.request.Request(f'{API}/{path}', headers={'Referer': 'https://judge.yosupo.jp/'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


@functools.lru_cache(maxsize=None)
def fetch(submission: int) -> Run:
    data = get(f'submissions/{submission}')
    overview, cases = data['overview'], data['case_results']
    return Run(submission, overview['problem_name'], overview['user_name'],
               hashlib.sha256(data['source'].encode()).hexdigest()[:16],
               {c['case']: c['time'] * 1000 for c in cases}, {c['case']: c['memory'] for c in cases})


def other_runs(run: Run) -> list[Run]:
    """The problem's other AC submissions by USER. Never another user's."""
    if run.user != USER:
        return []
    query = urllib.parse.urlencode({'problem': run.problem, 'user': USER, 'limit': 1000})
    ids = [s['id'] for s in get(f'submissions?{query}')['submissions'] if s['status'] == 'AC' and s['id'] != run.id]
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(fetch, ids))


def peers(run: Run, case: str) -> list[str]:
    def kind(name: str) -> str:
        return re.sub(r'_\d+$', '', name)
    same_class = [c for c in run.times if c != case and kind(c) == kind(case)]
    if len(same_class) >= 3:
        return same_class
    memory = run.memory[case]
    similar = [c for c in run.times if c != case and abs(run.memory[c] - memory) <= 0.05 * memory]
    return similar if len(similar) >= 3 else []


def matches(run: Run, other: Run, cases: list[str]) -> bool:
    gaps = [abs(run.times[c] - other.times[c]) for c in cases if c in other.times]
    return bool(gaps) and statistics.median(gaps) <= VERSION_GAP_MS


def reference(run: Run, case: str, others: list[Run]) -> float | None:
    """The time of case without a spike, or None if nothing tells."""
    same_source = [o.times[case] for o in others if o.source == run.source and case in o.times]
    if same_source:
        return statistics.median(same_source)
    case_peers = peers(run, case)
    versions = [o.times[case] for o in others if case in o.times and matches(run, o, case_peers)]
    return statistics.median(versions) if versions else None


def spikes(run: Run, others: list[Run]) -> dict[str, float]:
    """Spiked cases of run, each with its reference time."""
    found = {}
    for case, ms in run.times.items():
        ref = reference(run, case, others)
        if ref is not None and ref < NOISY_MS and SPIKE_MS[0] <= ms - ref <= SPIKE_MS[1]:
            found[case] = ref
    return found


def report(run: Run, others: list[Run]) -> str:
    spiked = spikes(run, others)
    base = {case: spiked.get(case, ms) for case, ms in run.times.items()}
    clean = max(base.values())
    exposed = sum(ms + SPIKE_SIZE_MS > clean for ms in base.values())
    judged = max(run.times.values())
    lines = [f'{run.id}: judged {judged:.0f} ms, clean {clean:.0f} ms; '
             f'P(clean run) = {(1 - SPIKE_RATE) ** exposed:.2f} ({exposed} cases within {SPIKE_SIZE_MS:.0f} ms of the max)']
    lines += [f'  spike: {case} {run.times[case]:.0f} (peers {ref:.0f})' for case, ref in spiked.items()]
    return '\n'.join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        run = fetch(int(arg))
        print(report(run, other_runs(run)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
