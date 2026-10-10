# pow_of_formal_power_series

N <= 500000 coefficients of f mod 998244353 and 0 <= M <= 10^18; print the first N coefficients
of f^M. 10 s. Largest tests: max_random, binary_exp_max, lower_deg_zero2_00 (f[0] != 0 or few
leading zeros, so u^M has close to 500000 coefficients; transforms up to 2^19).

Best judged: ours, 29 ms: [409299](https://judge.yosupo.jp/submission/409299) (`main.cpp` of #137);
#140 also 29 ms ([409311](https://judge.yosupo.jp/submission/409311),
[409312](https://judge.yosupo.jp/submission/409312)). #158: [409328](https://judge.yosupo.jp/submission/409328)
32 ms with launch spikes, clean 27 ms.
Record when opened (issue #65): 52 ms.

## Design

- f = x^k f_k u with u[0] = 1: f^M = 0 mod x^N if kM >= N (M > (N - 1) / k, no overflow);
  f^0 = 1 (also for f = 0); else f^M = x^(kM) f_k^M u^M with u^M = exp((M mod P) log u) of
  N - kM coefficients and f_k^(M mod (P - 1)).
- `lib/poly/pow.hpp`: `power(t, f, e, c, g)` = c (f / f[0])^e: d = e f'/f by log's blocked
  division, then exp's Newton steps on g' = d g from g[0] = c (lib/poly/notes.md). Computed in
  place at b[k, k + N - kM), then moved to b[kM, N) if kM != k.
- `lib/io` input; output in 10-byte fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for f, the text,
  the tables and the scratch. The program runs from `.preinit_array` and ends with `_exit`.

## Floor

`lc-amd`, max_random_00, whole process (`tools/runner.c`, judge flags, 21 interleaved runs):
read and write only (main.cpp without the power call) 4.5 ms; main.cpp 28.6 ms (median of the
first 8 rounds; later rounds of the same run took 33-51 ms, see Measuring). So power takes ~24
ms, about log (10.2) plus exp (13.5).

## Measuring

`lc-amd` (2026-10-09) slows down under sustained load, inside docker or not: in one container
run, 8 rounds at 28.6 ms, then 13 at 50.5 ms; exp's main.cpp went from 18.1 to 22-28 ms over 30
runs; on the host (`taskset -c 3`, 40 runs) 29.0 ms first, then 32-33, then 35-41. Not free
memory (dropping the page cache did not help), not other processes (none), not huge-page
fallbacks (6 huge-page faults per run, no fallbacks). Absolute times from long runs are
unreliable; same-run ratios (`judge.py bench`) still hold.

## Log

- 2026-10-09, claude (round 1): first solution, with `lib/poly/pow.hpp` (issue #95).
  - power = f / f[0] into g, `log` in place, times e, `exp` in place, times c.
  - Checks: 37/37 official tests (`tools/judge.py test`, `lc-amd`, slowest 28.7 ms:
    max_random_00, binary_exp_max_00, lower_deg_zero2_00); `stress.py` 400 rounds (N <= 3000
    against `brute.cpp`, a Miller recurrence; larger N by the recurrence at random coefficients;
    M in {0, 1, 2, 3, P - 1, P, 2P, multiples of P and P - 1, 2^59 - 1, 10^18, random}, leading
    zeros, monomials, f = 0); ASan/UBSan on 15 official cases, file and pipe input; lib/poly
    tests at -O2 (x86-64-v3 and native) and ASan/UBSan (`lc-intel`). Mutations caught:
    exponent e + 1 (lib test), f_k^(M mod P) for f_k^(M mod (P - 1)) (stress).
  - Merged as #137. Submitted its `main.cpp`: [409299](https://judge.yosupo.jp/submission/409299)
    AC 29 ms, 17.3 MiB.
  - Fused: `power` computes d = e f'/f with log's blocked division (`detail::log_derivative`,
    any f[0]) and runs exp's Newton steps (`detail::exp_newton`) from exp_direct times c. Saves
    log's integral, exp's derivative and three scaling passes. solution.cpp skips the move when
    kM = k.
  - In process at N = 500000 (`lc-amd`, warm): log_derivative 9.93 ms, exp_newton 13.34 ms,
    total 23.27 (23.89 on first use).
  - `judge.py bench` fused / #137 (21 rounds, slowest 3): `lc-amd` 28.87 / 29.34 ms, 0.9827;
    `lc-intel` 31.02 / 31.41 ms, 0.9885. exp and log re-bundled: exp 0.9994, 0.9954; log
    1.0000, 0.9988 (lib/poly/notes.md).
  - Checks: 37/37 official tests (slowest 28.5 ms); `stress.py` 400 rounds; lib/poly tests at
    -O2 (x86-64-v3, native) and ASan/UBSan (`lc-intel`); exp 26/26, log 25/25 official tests.
  - Merged as #140 (CI: pow 0.9872, exp 1.0026, log 0.9995; all 0.9964). Submitted its
    `main.cpp`: [409311](https://judge.yosupo.jp/submission/409311) AC 29 ms, 19.5 MiB;
    [409312](https://judge.yosupo.jp/submission/409312) AC 29 ms, 19.3 MiB. The ~0.4 ms gain
    is below the judge's 1 ms resolution.
- 2026-10-09, claude (lib/poly round, issue #95): faster leaf products (#158; lib/poly/notes.md).
  `judge.py bench` (21 rounds): `lc-amd` 29.33 -> 27.33 ms (0.9311), `lc-intel` 0.9768; CI
  0.9402. Submitted the merged `main.cpp`: [409328](https://judge.yosupo.jp/submission/409328)
  AC 32 ms, 19.0 MiB: launch spikes on random_00 and monomial_ans_low_deg_03 (32 ms each); the
  large cases 25-27 ms against 28-29 in 409312, so the clean score is 27 ms.
- Next: exp's T_m(x q) at the last two steps from the log's stored T(q_0), T(q_1) (~0.4 ms,
  lib/poly/notes.md); the transform levels (lib/ntt's kernels) are the largest shared cost.
