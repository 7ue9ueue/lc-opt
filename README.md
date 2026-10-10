# lc-opt

Solve every [Library Checker](https://judge.yosupo.jp) problem, then make each solution as fast as we can.
The code is written by AI agents. A human sets up the environment, writes the rules and keeps everything running.

Account: [Aiyiyi](https://judge.yosupo.jp/user/Aiyiyi).

## Results

Every problem we have finished beats the fastest time on the judge before us.

| Problem | Ours | Fastest before us |
|---|---:|---:|
| [many_aplusb](problems/sample/many_aplusb) | [18 ms](https://judge.yosupo.jp/submission/409103) | 23 ms |
| [convolution_mod](problems/convolution/convolution_mod) | [13 ms](https://judge.yosupo.jp/submission/409226) | 23 ms |
| [convolution_mod_large](problems/convolution/convolution_mod_large) | [430 ms](https://judge.yosupo.jp/submission/409343) | 737 ms |
| [convolution_mod_1000000007](problems/convolution/convolution_mod_1000000007) | [23 ms](https://judge.yosupo.jp/submission/409310) | 29 ms |
| [convolution_mod_2_64](problems/convolution/convolution_mod_2_64) | [43 ms](https://judge.yosupo.jp/submission/409296) | 76 ms |
| [convolution_F_2_64](problems/convolution/convolution_F_2_64) | [42 ms](https://judge.yosupo.jp/submission/409339) | 409 ms |
| [bitwise_and_convolution](problems/convolution/bitwise_and_convolution) | [12 ms](https://judge.yosupo.jp/submission/409202) | 26 ms |
| [bitwise_xor_convolution](problems/convolution/bitwise_xor_convolution) | [14 ms](https://judge.yosupo.jp/submission/409210) | 25 ms |
| [gcd_convolution](problems/convolution/gcd_convolution) | [15 ms](https://judge.yosupo.jp/submission/409214) | 37 ms |
| [lcm_convolution](problems/convolution/lcm_convolution) | [16 ms](https://judge.yosupo.jp/submission/409237) | 37 ms |
| [mul_mod2n_convolution](problems/convolution/mul_mod2n_convolution) | [21 ms](https://judge.yosupo.jp/submission/409247) | 81 ms |
| [mul_modp_convolution](problems/convolution/mul_modp_convolution) | [12 ms](https://judge.yosupo.jp/submission/409289) | 45 ms |
| [min_plus_convolution_convex_convex](problems/convolution/min_plus_convolution_convex_convex) | [13 ms](https://judge.yosupo.jp/submission/409301) | 20 ms |
| [min_plus_convolution_convex_arbitrary](problems/convolution/min_plus_convolution_convex_arbitrary) | [11 ms](https://judge.yosupo.jp/submission/409229) | 38 ms |
| [min_plus_convolution_concave_arbitrary](problems/convolution/min_plus_convolution_concave_arbitrary) | [27 ms](https://judge.yosupo.jp/submission/409239) | 117 ms |
| [multivariate_convolution](problems/convolution/multivariate_convolution) | [14 ms](https://judge.yosupo.jp/submission/409322) | 117 ms |
| [multivariate_convolution_cyclic](problems/convolution/multivariate_convolution_cyclic) | [12 ms](https://judge.yosupo.jp/submission/409321) | 117 ms |

Times are the judge's, for the slowest test, checked against the judge's API on 2026-10-10.
Each time links to its submission.

Open problems: [issues](https://github.com/7ue9ueue/lc-opt/issues).

## How it works

**The human side.** One person, one Claude Max plan ($200/month, the 20x tier), about five agents at a time.
The human built the workbench and the agents use it:

- **Rules.** [`AGENTS.md`](AGENTS.md) sets the standards: code style, how to measure, what counts as a win.
- **Machines.** Two Google Cloud VMs. `lc-amd` has the judge's exact CPU (AMD EPYC 7B13, Zen 3) and gives
  judge-like timings. `lc-intel` has hardware counters for `perf` profiling.
- **A copy of the judge.** [`tools/judge.py`](tools/judge.py) uses the judge's compiler image, flags,
  official tests and checker. Its timings match the judge to within 1 ms.
- **CI.** Every pull request runs all official tests and times the new code against `main` on 3 GitHub
  machines. It merges by itself only if it is correct and not slower.
- **Submission.** [`tools/submit.py`](tools/submit.py) submits to the judge, at most 5 times per version.

**The agent side.** Each problem is a GitHub issue. `/work` ([`.claude/commands/work.md`](.claude/commands/work.md))
starts one subagent per problem, briefed with [`tools/prompt.md`](tools/prompt.md). A round is:
read the notes, try an idea, measure it, log it, open a pull request. CI decides. A problem gets at most
5 rounds and stops after 2 rounds in a row without gain.

The rounds usually go: pick the right algorithm, fix memory and I/O, then AVX2, then assembly where the
profile shows the compiler is the limit.

## Origin

This continues [QPoly](https://github.com/7ue9ueue/SymPoly), a hand-written AVX2 NTT project.
Its kernels were rewritten into [`lib/ntt`](lib/ntt) and now power the convolution problems.

## Rules in brief

- We write all code ourselves. Papers and other code may be read for ideas, never copied. Sources are
  cited in each `notes.md`.
- We never look at other users' submissions. Leaderboard times only.
- No speed claim without passing tests and a measurement.
- At most 5 judge submissions per session.

## Layout

```
AGENTS.md                    rules for agents (and humans)
lib/                         shared code: fast I/O, NTT
problems/<category>/<name>/
  solution.cpp               source, includes lib/
  main.cpp                   bundled single file, what we submit
  brute.cpp                  slow reference for stress tests
  notes.md                   research and every attempt, won or lost
tools/                       judge copy, CI verdict, submission, round brief
```

## Build and test

The tools need Linux and Docker.

```bash
python3 tools/bundle.py problems/sample/many_aplusb/solution.cpp
python3 tools/judge.py test many_aplusb problems/sample/many_aplusb/main.cpp
python3 tools/judge.py bench many_aplusb old.cpp new.cpp
```

`bundle.py` writes `main.cpp`. `test` runs every official test with the official checker.
`bench` times two versions in the same run; a ratio below 1 means the new one is faster.
