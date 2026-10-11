# Luogu P5383 普通多项式转下降幂多项式

https://www.luogu.com.cn/problem/P5383

- Given a_0..a_(n-1), find b with sum b_i x^(i falling) = sum a_i x^i, mod 998244353.
- 10 tests: 3 with n = 2000, 7 with n = 10^5. a_i in [0, P). TL 2 s, ML 500 MiB (512000 KB).
- Output: one line, b_0..b_(n-1).

## Method

- sum_i F(i) x^i / i! = (sum_k b_k x^k) e^x, so b = (sum_i F(i) x^i / i!) e^(-x) mod x^n.
- Intended: multipoint evaluation of F at 0..n-1, O(n log^2 n), then one product. The
  constant of the evaluation decides the verdict.
- Our route (`solution.cpp`): the textbook remainder tree. Subproduct tree of (x - i), leaves of
  at most 64 points. Descent: F mod each child's product, by q = rev(rev(F) / rev(T)) (one
  `easy::inverse`, one `easy::multiply`) and r = F - q T mod x^deg T (one more product). Leaves by
  Horner, 64 n operations. About 4100 inverses and 8200 products.
- Brute (`brute.cpp`): Horner at 0..n-1, then b_k = (forward difference)^k F(0) / k!, O(n^2).

## Evidence that the route is tight

- Luogu forum threads on P5383 (titles given by the coordinator; the forum needs a login, not
  re-checked here): "您们太强了，我自己放的题stdTLE了怎么办？" (the setter's reference solution
  exceeded the limit) and "求开大时限" (a request to raise the limit).
- Our textbook-NTT baseline, same algorithm: 1.53 s CPU on lc-amd, 77% of the 2 s limit, on a
  4.4 GHz Zen 3. Luogu machines are likely slower (guess).

## Times

lc-amd (EPYC 7B13), gcc:13 `-std=c++20 -O2`, each run under `flock /tmp/bench.lock`, 2026-10-10.
max0: n = 10^5 random; max1: n = 10^5, all a_i = P - 1. Raw: `lc-opt-explore/showcase-p5383/time*.txt`.

| Program | max0 | max1 | user CPU |
|---|---|---|---|
| `solution.cpp` (remainder tree, easy::inverse) | 432 ms median (318-446) | 307 ms (295-468) | 0.12 s, plus 0.30 s sys |
| `baseline.cpp` (same, textbook NTT) | 1542 ms (1093-1625) | 1605 ms (1506-1645) | 1.53 s |
| variant: Newton inverse on easy::multiply | 110 ms | 107 ms | 0.15 s |
| variant: easy::evaluate (transposed tree) | 33 ms | 24 ms | 0.02 s |

- 70% of our time is system time: each `easy::inverse` call maps a fresh arena with
  `MADV_HUGEPAGE`, and lc-amd has THP `defrag=madvise`, so every call faults in (and may
  compact for) a 2 MiB page. 4100 calls, ~70 us each. The variant that builds the inverse from
  `easy::multiply` (one reused workspace) has no such cost. Luogu's THP setting is unknown.
- The machine was loaded during these runs (real > user + sys for the baseline at times).
- `showcase/check.py`'s timing includes the wait for the flock, so its numbers under contention
  are too high (it reported 1507 ms median for max0 while runs inside the lock took 0.43 s).

## Correctness

- Sample OK. check.py stress: 300 small cases against brute (n from 1 to 2000; all zeros; all P - 1).
- ASan + UBSan (gcc:13 -O1): 41 small cases and max0, clean and correct.
- Baseline agrees with brute on 41 small cases and with solution on both max cases.

## Log

- 2026-10-10, showcase agent: built the remainder-tree solution, baseline, both variants. Shipped
  the remainder tree with easy::inverse: 0.43 s against 2 s.
