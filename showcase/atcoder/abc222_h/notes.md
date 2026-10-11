# ABC222 H Beautiful Binary Tree

https://atcoder.jp/contests/abc222/tasks/abc222_h. N <= 10^7, TL 3 s, 1024 MiB, mod 998244353.

## Problem and routes

- Answer: [x^(N-1)] (1 + 3x + x^2)^(2N) / N (Lagrange inversion on A = x (1 + 3A + A^2)^2).
- Intended: O(N) from the coefficients' P-recursive recurrence.
- Editorial (https://atcoder.jp/contests/abc222/editorial/2742) on the O(N log N) series route:
  「これでも TL には間に合いません」 ("even this doesn't make the TL").
- Ours: that route. `easy::pow` of f = 1 + 3x + x^2 to the power 2N mod x^N: exp(2N log f),
  O(N log N), transforms up to 2^24.

## Results

2026-10-10, claude. lc-amd (EPYC 7B13), gcc:15.2.0 `-std=gnu++23 -O2 -march=native`
(AtCoder's flags), `showcase/check.py`, 5 runs each.

| Input | solution.cpp | baseline.cpp |
|---|---:|---:|
| N = 10^7 | 630 ms median (642 max) | 65.5-75.7 s (3 runs), 666 MiB |
| N = 8388609 | 537 ms median | - |
| N = 10^6 | - | 3.15 s (3 runs) |

- Samples (1, 2, 222, 222222) and 100 random N against brute.cpp (the recurrence): OK.
- baseline.cpp: the same exp(2N log f) with a textbook NTT (iterative radix-2, bit reversal,
  `% P`) and plain Newton iterations (exp recomputes log, log recomputes the inverse each step).
  It already misses the TL at N = 10^6. A tuned contest NTT would be several times faster than
  this baseline (guess), still far over 3 s at 10^7.
- Judge caveat: AtCoder's CPU model is not published. `-march=native` there; the pragma in
  lib/easy enables AVX2 if the build does not.
