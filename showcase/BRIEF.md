# Showcase round brief

Goal: show that our NTT library makes a *brute-force* NTT solution pass where it normally times
out. Per problem: a solution that uses the simple, heavier NTT route (an extra log factor, many
convolutions, divide and conquer instead of the clever method), built on `lib/easy`, that passes
the real limits; and a baseline with the *same algorithm* on a plain textbook NTT, to show the
gap. Read `AGENTS.md` (rules, writing style) and `lib/easy/poly.hpp` (the API) first.

## Files: `showcase/<judge>/<problem>/`

- `solution.cpp`: the solution. Standard headers first, then `lib/easy/io.hpp` and
  `lib/easy/poly.hpp` (the pragma in poly.hpp must come after the standard headers). Use
  `easy::Reader`/`easy::Writer` for I/O (portable; `lib/io` is Linux-only). A short header
  comment: problem, limits, the algorithm and its complexity, why it is the brute-force route.
- `main.cpp`: `python3 showcase/bundle.py showcase/<judge>/<problem>/solution.cpp`. Never edit by hand.
- `baseline.cpp`: the same algorithm with a textbook NTT you write yourself (iterative radix-2,
  bit reversal, `% P` or Montgomery scalar code, roughly what a typical contestant writes; not
  deliberately slow). Self-contained, no lib/ includes.
- `brute.cpp`: a simple, obviously correct reference (small inputs only).
- `gen.py`: `gen.py SEED small` prints a small random input; `gen.py SEED max` a largest-size
  input (the worst cases for our solution; several seeds if shape matters); `gen.py --max-cases`
  prints how many max seeds there are. Include edge cases among small seeds.
- `samples/*.in`, `samples/*.out`: the statement's samples.
- `check.py` (optional): `check.py IN OUT ANSWER`, exit 0 if OUT is right, when answers are not unique.
- `notes.md`: problem link, limits, intended solution, our route and complexity, evidence that
  the route normally fails (links, short quotes), measured times (ours, baseline, both on max
  cases, on lc-amd), and any judge caveats. A log, not an essay.

Do not edit `lib/`, `tools/`, `showcase/check.py` or `showcase/bundle.py`. If `lib/easy` lacks
something you need, write it locally in solution.cpp, and say so in your report (it may move
into lib/easy later). Do not commit or push; the coordinator commits.

## Testing on the VM

Work in your own copy of the worktree on `lc-amd` (other agents share the VM):

```
W=/Users/aiyiyi/Documents/cpp_hpc/lc-opt/.claude/worktrees/ntt-library-problems-633247
COPYFILE_DISABLE=1 tar czf - --exclude=.git --exclude='._*' -C $W . | gcloud compute ssh lc-amd --zone=europe-west2-b --command 'rm -rf ~/ws/<problem> && mkdir -p ~/ws/<problem> && tar xzf - -C ~/ws/<problem>'
gcloud compute ssh lc-amd --zone=europe-west2-b --command 'cd ~/ws/<problem> && python3 showcase/check.py showcase/<judge>/<problem>'
```

`check.py` builds with the judge's compiler and flags, runs samples, stress-tests against
brute.cpp, runs the Windows build under Wine for Codeforces, and times the max cases (under
`/tmp/bench.lock`, held for all runs). Time `baseline.cpp` the same way (build it with the same
docker image and flags; take the lock once around all runs, e.g. `flock /tmp/bench.lock sh -c '...'`,
so waiting for the lock is not timed). Also run solution.cpp once under
`-fsanitize=address,undefined` on small cases.

Judges: AtCoder GCC 15.2 Linux with -march=native; Codeforces GCC 14.2 on Windows, -O2, no
-march (the pragma gives AVX2); Luogu, LOJ, QOJ, CodeChef: Linux GCC, assume 13, -O2, no -march.
Judge machines may be slower than lc-amd; aim for a margin of at least 2x under the limit.

Keep exploration files (variants, scripts, results) in `~/explore/showcase-<problem>/` on the VM
and copy them to `~/Documents/cpp_hpc/lc-opt-explore/showcase-<problem>/` on the Mac before you
finish. Never in /tmp (binaries and generated inputs excepted).

Problem statements: codeforces.com and its mirror refuse fetches from here. Luogu mirrors CF and
AtCoder problems (`https://www.luogu.com.cn/problem/CF986D`; the statement is JSON in the page).
QOJ statements are mirrored at jiang.ly. Read editorials and blogs freely; never open or copy
anyone's submission code.

## Report

Lead with the result: passes or not, our max-case time, the baseline's, the time limit. Then
what you built, anything uncertain (statement details, judge flags), and lib/easy requests.
