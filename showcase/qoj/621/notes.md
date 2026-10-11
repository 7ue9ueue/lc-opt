# QOJ 621 多项式指数函数

https://qoj.ac/problem/621 (statement read from the mirror https://jiang.ly/problem/621).

- Input: n, then a_0..a_(n-1) with a_0 = 0. Output: exp(A) mod z^n, mod 998244353, on one line.
- 1 <= n <= 10^6, 0 <= a_i < 998244353. 10 subtasks, n <= 100, 5e3, 3e4, 1e5, 1.5e5, 2e5, 3e5, 5e5, 7e5, 1e6.
- 2.5 s, 1024 MB. Judge: Linux GCC, flags assumed `-std=c++20 -O2`, no `-march` (guess; not stated).

## Route

- Intended: Newton iteration, O(n log n).
- Ours: g' = f' g gives i g_i = sum_(k=1..i) k f_k g_(i-k). Fill g online by divide and conquer
  (CDQ): solve [l, m), add (g[l..m) * h[0..r-l)) to the pending sums of [m, r) with one
  `easy::multiply`, solve [m, r). Ranges of <= 64 run the quadratic recurrence. O(n log^2 n).
- Each product is a full linear product of sizes s/2 and s, so its transform has length
  bit_ceil(1.5 s) (2^21 at the top for n = 10^6). A cost model (sum of L log L over nodes) puts
  n = 10^6 at the maximum over all n <= 10^6.
- The OI Wiki ln/exp page lists both the D&C and the Newton method. That the D&C route fails
  the TL is our measurement below, not a quote.

## Measurements (2026-10-10)

QOJ flags: gcc:13 docker image, `-std=c++20 -O2`. Max cases: `gen.py 0 max` (n = 10^6, random),
`gen.py 1 max` (n = 10^6, all a_i = P - 1). Interleaved runs, wall time of the whole process.

lc-bench (EPYC 7B13, idle, one hold of `flock /tmp/bench.lock` around all runs), 9 runs (5 for cpalgo):

| program | max0 median | max1 median | spread |
|---|---|---|---|
| I/O floor (read, write n numbers) | 52 ms | 50 ms | 50-54 |
| `easy::exp` (Newton, reference only) | 81 ms | 80 ms | 80-95 |
| solution (D&C on lib/easy) | 224 ms | 223 ms | 223-225 |
| baseline.cpp (D&C, textbook NTT, root table) | 1711 ms | 1711 ms | 1708-1747 |
| cpalgo variant (D&C, w *= w_len per butterfly) | 2840 ms | 2842 ms | 2840-2866 |

lc-amd (same CPU, load average 5.8-6.4 on 4 vCPUs from other agents), 7 runs, medians:
floor 89/87, Newton 144/127, solution 378/366, baseline 2739/2668, cpalgo 3865/3489 ms.
Raw: `lc-opt-explore/showcase-qoj621/bench-lc-*.txt`.

- Solution: 224 ms, 11x under the TL. Baseline: 1711 ms, 0.68 of the TL on an idle EPYC 7B13;
  7.6x slower than ours.
- The baseline passes on lc-bench without a 2x margin. The cpalgo variant (the common
  cp-algorithms-style loop, two `% P` per butterfly) fails at 2.84 s. Whether the baseline passes
  on QOJ depends on its judge speed (unknown).

## Checks

- Sample: OK. `showcase/check.py --only stress`: 300 cases OK against brute.cpp.
- ASan/UBSan (gcc:13, `-O1 -fsanitize=address,undefined`): 60 small seeds and n = 5000 (all P - 1), clean.
- All four programs agree on both max cases.
- `showcase/check.py` (full run, lc-amd at load average 11): samples OK, 300 stress cases OK,
  max cases 374 ms and 368 ms median.
- Rebundled after the lib/easy cache change (commit 4f361bd); re-timed on lc-bench: 224 ms, unchanged.

## Files

Exploration: `~/Documents/cpp_hpc/lc-opt-explore/showcase-qoj621/` (newton_exp.cpp, io_floor.cpp,
baseline_cpalgo.cpp, time.sh, bench.py, results).
