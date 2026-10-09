# lc-opt

Solve every [Library Checker](https://judge.yosupo.jp) problem, then make each solution as fast as we can.

## Hard rules

- Never open, download, or run another user's Library Checker submission. Leaderboard times are fine.
- Write all code ourselves. Papers, docs, and other open-source code may be read for ideas, never copied.
  Cite what you used in `notes.md`.
- QPoly kernels (`../SymPoly/work/`) may be reused, but only after a rewrite to this repo's standards.
  Never use `study/` or vendored files.
- Submit to Library Checker at most 5 times per problem, and only versions that passed CI.
  Record each submission ID and judged time in `notes.md`. Never commit credentials.
- No speed claim without passing checks and a measurement.

## Writing

- English. Short, plain sentences. No filler, no restating.
- Numbers over adjectives. Mark guesses as guesses.
- Notes are logs, not essays: what was tried, the result, a link to evidence.
- Replies to the user lead with the result.
- Commits: imperative one-liner, at most 72 characters.

## Code

- C++23, GCC 15.2. Each submission is one self-contained file, bundled from `lib/`.
- Professional quality. Readable first: clear names, small functions, no dead code.
- `lib/` modules have a small, documented API and their own tests.
- Comments are short and rare: invariants, value ranges, overflow bounds, memory layout.
  No banner or essay comments.
- Delete losing variants once they are logged. Git keeps the history.

## Target

- Judge: AMD EPYC 7B13 (Zen 3), one core, 1 GiB.
- Compile: `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native main.cpp`.
- Time covers the whole process: start, I/O, page faults, exit. The score is the slowest case.
- Aim for code that is fast on AMD Zen in general. AVX2 at most; no AVX-512.
- Judge-specific tricks (e.g. padded output the checker accepts) stay in the problem's `main.cpp`, never in `lib/`.

## Machines

| Machine | Use |
|---|---|
| Mac (ARM64) | Edit, build, correctness. x86 binaries run under Rosetta; never time them. |
| GCP `lc-intel` (c4-standard-4, Xeon 8581C, PMU on) | Profiling with `perf`: core events and top-down. No L3 events. |
| GCP `lc-amd` (c2d-standard-4, EPYC 7B13, the judge's CPU) | Judge-like timing. No hardware counters. |
| GitHub Actions (EPYC 7763 Zen 3, plus other CPUs) | Timing. Confirm wins on the judge's core without losses elsewhere. |

- VMs are in project `project-c73e6eb1-e167-4d7a-a31`, region `europe-west2` (London). Reach them with
  `gcloud compute ssh <name> --zone=<zone>`. Both run Ubuntu 24.04 with Docker and the pinned `gcc:15.2.0` image.
- Off the judge, build with `-march=x86-64-v3` (AVX2, no AVX-512). On `lc-amd`, use the judge's exact command.
- On a VM, wrap every timing or profiling run in `flock /tmp/bench.lock`. Builds and tests may run in parallel.
- Keep VMs running; do not stop them.
- Need more capacity? Create a VM yourself: same project and region, Ubuntu 24.04, name `lc-<purpose>`.
  Add it to this table. Never delete a VM you did not create.

## Layout

```
AGENTS.md                    rules (CLAUDE.md imports it)
STATUS.md                    foundations and one row per problem
lib/                         shared code: I/O, modint, NTT, ...
problems/<category>/<name>/  category as in library-checker-problems
  main.cpp                   current best submission
  brute.cpp                  simple reference for stress tests
  notes.md                   research, ideas, attempt log
```

## Process

Each problem goes through four passes. Several agents repeat a pass until a round gives no measurable gain.

0. **Research**: algorithms, papers, limits. Measure the floor (read input, write output, nothing else)
   and compare it with the record to estimate headroom.
1. **High level**: algorithm, data layout, memory, I/O. Portable C++.
2. **SIMD**: AVX2 kernels. For pointer-heavy problems: memory layout, prefetching, branch-free code.
3. **Assembly**: inline asm and instruction scheduling, only where profiling shows compiled code is the limit.

Stop when the solution sits at the floor or leads the record by more than noise.
A change to `lib/` is re-checked on every problem that uses it.

## Working in parallel

- One agent per problem at a time. Each agent uses its own git worktree and branch: `<agent>/<problem>`.
- Read `notes.md` first. Do not repeat a logged attempt without a new reason.
- Log every attempt, win or loss: date, agent, idea, result, evidence.
- `main.cpp` must stay a standalone file that can be submitted as is. It may be generated
  (e.g. by a Python script or from several files); commit the generator and the generated file.
- Open a pull request per round. It merges automatically once CI confirms correctness and no slowdown.

## Correctness

- All official tests pass: generated from `yosupo06/library-checker-problems` at a pinned commit,
  judged by the problem's checker.
- Stress-test against `brute.cpp` with random small inputs and edge cases (min/max sizes, zeros, max values).
- The ASan/UBSan build passes before promotion.

## Measuring

- Time baseline and candidate in the same run: same inputs, interleaved, repeated. Report median and spread.
- Differences within noise are inconclusive.
- Record CPU, compiler, flags, commit, and raw numbers.

## Status

After a meaningful result, update the problem's `notes.md` and its `STATUS.md` row:
pass, best time (estimated or judged), record, next idea.
