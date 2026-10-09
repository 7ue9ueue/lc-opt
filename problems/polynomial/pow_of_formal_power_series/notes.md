# pow_of_formal_power_series

N <= 500000 coefficients of f mod 998244353 and 0 <= M <= 10^18; print the first N coefficients
of f^M. 10 s. Largest tests: max_random, binary_exp_max, lower_deg_zero2_00 (f[0] != 0 or few
leading zeros, so u^M has close to 500000 coefficients; transforms up to 2^19).

Best judged: none yet.
Record when opened (issue #65): 52 ms.

## Design

- f = x^k f_k u with u[0] = 1: f^M = 0 mod x^N if kM >= N (M > (N - 1) / k, no overflow);
  f^0 = 1 (also for f = 0); else f^M = x^(kM) f_k^M u^M with u^M = exp((M mod P) log u) of
  N - kM coefficients and f_k^(M mod (P - 1)).
- `lib/poly/pow.hpp`: `power(t, f, e, c, g)` = c (f / f[0])^e, here log then exp of lib/poly
  (lib/poly/notes.md). Computed in place at b[k, k + N - kM), then moved to b[kM, N).
- `lib/io` input; output in 10-byte fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for f, the text,
  the tables and the scratch. The program runs from `.preinit_array` and ends with `_exit`.

## Floor

`lc-amd`, max_random_00, whole process (`tools/runner.c`, judge flags, 21 interleaved runs):
read and write only (main.cpp without the power call) 4.5 ms; main.cpp 28.6 ms (median of the
first 8 rounds; later rounds of the same run took 33-51 ms, see Measuring). So power takes ~24
ms, about log (10.2) plus exp (13.5).

## Measuring

On `lc-amd` (2026-10-09), runs inside one long-lived container (judge image, `--memory 1g`,
`--cpuset-cpus 3`) drifted: 8 rounds at 28.6 ms, then 13 at 50.5 ms; exp's main.cpp went from
18.1 to 22-28 ms over 30 runs. Dropping the page cache did not help; no other process ran.
Outside docker (`taskset -c 3`, the same binary, 16 runs): 28.6-30.1 ms, 6 huge-page faults
and ~775 page faults per run, no fallbacks. Cause unknown; absolute times from long container
runs are unreliable, same-run ratios (`judge.py bench`) still hold.

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
