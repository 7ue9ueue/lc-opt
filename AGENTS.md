# lc-opt

Solve every [Library Checker](https://judge.yosupo.jp) problem, then make each solution as fast as we can.

## Hard rules

- Never open, download, or run another user's Library Checker submission. Leaderboard times are fine.
- Write all code ourselves. Papers, docs, and other open-source code may be read for ideas, never copied.
  Cite what you used in `notes.md`.
- QPoly kernels (`../SymPoly/work/`) may be reused, but only after a rewrite to this repo's standards.
  Never use `study/` or vendored files.
- Submit to Library Checker at most 5 times per session, and only versions that passed CI.
  If a session needs more, label the issue `blocked` and leave it to the user.
  Record each submission ID and judged time in `notes.md`. Never commit credentials.
- No speed claim without passing checks and a measurement.

## Writing

- English. Short, plain sentences. No filler, no restating.
- Numbers over adjectives. Mark guesses as guesses.
- Notes are logs, not essays: what was tried, the result, a link to evidence.
- Replies to the user lead with the result.
- Commits: imperative one-liner, at most 72 characters.

## Code

- C++23, GCC 15.2. Each submission is one self-contained file, bundled from `lib/`:
  write `solution.cpp` with `#include "lib/..."`, then `python3 tools/bundle.py <dir>/solution.cpp`
  writes `main.cpp`. Commit both; CI checks that they match.
- Use `lib/io` for all input and output (`io::Reader`, `io::Writer`; see the header of `lib/io/io.hpp`).
- Professional quality. Readable first: clear names, small functions, no dead code.
- `lib/` modules have a small, documented API and their own tests.
- Comments are short and rare: invariants, value ranges, overflow bounds, memory layout.
  No banner or essay comments.
- Delete losing variants from the repo once they are logged. Their files stay in the exploration folder.

## Target

- Library Checker account: **Aiyiyi** (https://judge.yosupo.jp/user/Aiyiyi). All submissions go under it.
- Judge: AMD EPYC 7B13 (Zen 3), one core, 1 GiB.
- Compile: `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native main.cpp`.
- Time covers the whole process: start, I/O, page faults, exit. The score is the slowest case.
- Aim for code that is fast on AMD Zen in general. Vectors are AVX2 at most; see Instruction sets.
- Judge-specific tricks (e.g. padded output the checker accepts) stay in the problem's `main.cpp`, never in `lib/`.

## Instruction sets

Checked on the judge with `tools/isa_probe.cpp` (aplusb, [409083](https://judge.yosupo.jp/submission/409083): AC) and on `lc-amd`.

- Vectors: AVX2, 16 ymm registers. AVX-512 instructions fault with SIGILL.
- `-march=native` also enables the extensions below. Use them where they fit.

| Extension | Instructions | Use |
|---|---|---|
| BMI1, BMI2, LZCNT, POPCNT | `tzcnt`, `lzcnt`, `popcnt`, `pdep`, `pext`, `mulx` | bitsets, rank/select, bit packing, 64-bit modmul |
| ADX | `adcx`, `adox` | two independent carry chains in big-integer multiply |
| PCLMUL, VPCLMULQDQ | carry-less multiply, xmm and ymm | GF(2) polynomials, mod-2 matrices |
| AES, VAES, SSE4.2 | `aesenc` (xmm, ymm), `crc32` | hash mixing |

- Zen 3 costs, measured on `lc-amd`: `pdep` and `pext` take 3 cycles latency, 1 per cycle.
  `vpgatherdd` costs 1.38 cycles per element against 0.90 for scalar loads (table in L1/L2);
  on `lc-intel` the gather wins, 0.58 against 0.71. Look up other costs on uops.info, then measure.
- Prefer intrinsics from `<immintrin.h>`. Use inline asm when no intrinsic yields the instruction:
  GCC 15 compiles `_addcarryx_u64` to `adc`, never `adcx`/`adox`.
- Guard each use with its macro (e.g. `#ifdef __VPCLMULQDQ__`), keep a portable fallback, and test both.
- `-march=x86-64-v3` lacks ADX, PCLMUL, VPCLMULQDQ, AES and VAES, so guarded code falls back there.
  To build the judge's path elsewhere, add `-madx -mpclmul -mvpclmulqdq -maes -mvaes` (`lc-intel` has them all).
- On an AVX-512 machine, `-march=native` lets GCC emit AVX-512 unasked (mask registers in a plain `-O2` loop
  on `lc-intel`). Timings of such a build measure code the judge cannot run.

## Machines

| Machine | Use |
|---|---|
| Mac (ARM64) | Edit, build, correctness. x86 binaries run under Rosetta; never time them. |
| GCP `lc-intel` (c4-standard-4, europe-west2-c, Xeon 8581C, PMU on) | Profiling with `perf`: core events and top-down. No L3 events. |
| GCP `lc-amd` (c2d-standard-4, europe-west2-b, EPYC 7B13, the judge's CPU) | Builds and tests. Same CPU and setup as `lc-bench`. No hardware counters. |
| GCP `lc-bench` (c2d-standard-4, europe-west2-b, EPYC 7B13, the judge's CPU) | Timing only, so builds and tests elsewhere do not disturb it. No hardware counters. |
| GCP `lc-k68` (c2d-standard-4, europe-west2-b, EPYC 7B13, Linux 6.8) | Judge-like kernel: timing of programs heavy in page faults, huge pages or tmpfs output. On convolution_mod_large it is within 2 ms of judged times for 409343 and 4-6 ms below them for 409657, where `lc-amd` (Linux 7.0) is 4-14% below. |
| GitHub Actions (EPYC 7763 Zen 3, plus other CPUs) | Timing. Confirm wins on the judge's core without losses elsewhere. |

- VMs are in project `project-c73e6eb1-e167-4d7a-a31`, region `europe-west2` (London). Reach them with
  `gcloud compute ssh <name> --zone=<zone>`. All run Ubuntu 24.04 with Docker and the pinned `gcc:15.2.0` image,
  except `lc-k68`: no Docker, so run static binaries built on `lc-amd` (`-static`).
- Off the judge, build with `-march=x86-64-v3` (AVX2, no AVX-512). On `lc-amd` and `lc-bench`, use the judge's
  exact command.
- On a VM, wrap every timing or profiling run in `flock /tmp/bench.lock`. Builds and tests may run in parallel.
- Keep VMs running; do not stop them.
- Need more capacity? Create a VM yourself: same project and region, Ubuntu 24.04, name `lc-<purpose>`.
  Add it to this table. Never delete a VM you did not create.

## Tools

They need Linux and Docker: run them on a VM or in CI, not on the Mac.

- `python3 tools/judge.py test <problem> <file.cpp>`: the judge's compiler and command, every official test,
  the official checker.
- `python3 tools/judge.py bench <problem> <old.cpp> <new.cpp>`: same-run timing on the slowest tests.
  A ratio below 1 means the new file is faster.
- `python3 tools/cases.py <problem>`: build the official tests only (cached in `~/.cache/lc-opt`).
- `python3 tools/submit.py <problem> <file.cpp>`: submit to the judge as Aiyiyi and wait for the verdict.
  Mac only; enforces the 5-per-version cap. If it says "not logged in", stop and ask the user.
- `python3 tools/spikes.py <submission>...`: flag the judge's +9 ms launch spikes (5% of cases, not ours;
  see `tools/spikes.md`) and print the clean score. Mac or anywhere.
- `python3 tools/speed.py bench` (VM, own checkout `~/lc-speed`) and `render` (anywhere): `SPEED.md`, judged
  and AMD/Intel times per problem; `bench/<problem>.md`, every test with an I/O floor. Read-only, no submissions.
- `tools/isa_probe.cpp`: submit as aplusb to re-check the judge's instruction set. AC means every check holds;
  otherwise the answer is off by a bitmask of the failed checks, listed on stderr.
- Every VM has the repo at `~/lc-opt`. Run `git fetch` there and check out your branch.
- `main` is protected. Every change, docs included, goes through a pull request; enable
  `gh pr merge --auto --squash`. CI (`.github/workflows/verify.yml`) tests each changed `main.cpp` and times it
  against `main` on 3 machines. It merges only if correct and not slower: geomean over machines, then over problems.

## Layout

```
AGENTS.md                    rules (CLAUDE.md imports it)
STATUS.md                    foundations (per-problem status is in GitHub issues)
SPEED.md                     dashboard: judged, best other, AMD, Intel (generated by tools/speed.py)
bench/<problem>.md           every test's time and I/O floor (generated by tools/speed.py)
tools/                       judge copy, CI verdict, submission, round brief (prompt.md)
.claude/commands/work.md     /work: run rounds on ready issues with subagents
lib/                         shared code: I/O, NTT, multimod, poly, mem, run
problems/<category>/<name>/  category as in library-checker-problems
  solution.cpp               source; includes lib/ headers
  main.cpp                   current best submission, bundled from solution.cpp
  brute.cpp                  simple reference for stress tests
  notes.md                   research, ideas, attempt log
```

## Process

Work happens in rounds: one subagent per round, started by `/work` and briefed with `tools/prompt.md`.
A problem gets at most 5 rounds and stops after 2 rounds in a row without gain.
The passes below are the usual order of what to try.

0. **Research**: algorithms, papers, limits. Measure the floor (read input, write output, nothing else);
   the gap between it and our time is the headroom.
1. **High level**: algorithm, data layout, memory, I/O. Portable C++.
2. **SIMD**: AVX2 kernels and the extensions under Instruction sets. For pointer-heavy problems: memory layout, prefetching, branch-free code.
3. **Assembly**: inline asm and instruction scheduling, only where profiling shows compiled code is the limit.

Stop early only when the solution sits at the floor. Leading the record is not a reason to stop;
the record is a reference point, not the target.
A change to `lib/` is re-checked on every problem that uses it.

## Multiple architectures

We target several CPUs at once. When different variants win on different CPUs, keep each variant
where it wins instead of settling on one compromise: aim for the frontier.

1. Select at compile time first, from measurements on every machine we have. `-march=native` reveals
   the CPU through macros such as `__znver3__`; on the judge this is free and exact.
2. Read hardware facts at run time (cache sizes, CPU features) when a macro cannot capture them.
3. Benchmark at run time only when 1 and 2 cannot decide. Time real work (the first blocks of the actual
   input), not a dry run. The cost counts toward the judged time, and CI blocks it if it does not pay off.

Test every variant, not just the one a machine picks (force each with a `-D` flag). Record per-CPU
measurements in `notes.md`.

## Working in parallel

- One agent per problem at a time. Each agent uses its own git worktree and branch: `<agent>/<problem>`.
- To change a file another agent may be working on, prefer a comment on that agent's issue over an edit.
  Edit it yourself only when the change cannot cause a merge conflict.
- Read `notes.md` first. Do not repeat a logged attempt without a new reason.
- Keep every exploration file (variants, probes, scripts, results). Never put one in `/tmp`: VMs wipe it.
  On a VM, work in `~/explore/<problem>/`. On the Mac, the folder is
  `~/Documents/cpp_hpc/lc-opt-explore/<problem>/`. Before a round ends, copy each VM folder there,
  into `<vm>/`. A VM can be lost; the Mac copy is the record. Only throwaway files (binaries,
  generated inputs, `perf` data) may go to `/tmp`.
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

- Each problem has one GitHub issue, labeled `ready`, `running`, `blocked` or `done`.
- After each round, comment on the issue: what you tried, the numbers, and a last line that is exactly
  `Result: gain` or `Result: no gain`. `/work` reads that line.
- Keep `notes.md` current: best judged time, record, next idea. Do not edit `STATUS.md` from a problem round.
