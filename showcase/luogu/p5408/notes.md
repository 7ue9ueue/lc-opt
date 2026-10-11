# Luogu P5408 第一类斯特林数·行

https://www.luogu.com.cn/problem/P5408

- Task: given n, print the unsigned Stirling numbers of the first kind [n, 0], ..., [n, n].
- Limits: 1 <= n < 262144 (20% of tests: n <= 1000). Modulus 167772161 = 5 2^25 + 1. 500 ms and
  512000 KB on each of 10 tests.
- Sample: `3` gives `0 2 3 1`.

## Intended solution

[n, i] = [x^i] x (x + 1) ... (x + n - 1). Double the rising factorial: from f(x) = x^(rising m),
f(x + m) is a Taylor shift (one convolution), and f(x) f(x + m) = x^(rising 2m). O(n log n).

## Our route

Divide and conquer over the n linear factors: split [lo, hi) in half, multiply the two halves'
products. Leaves of at most 32 factors expand directly. Products with a factor of at most 32
coefficients are schoolbook; the rest use `multimod::Transform` with the prime 167772161
(lib/easy is 998244353 only). O(n log^2 n): log n levels, each a convolution of total size n.

The lib/multimod use (a local `Multiplier` with fixed buffers for 2^18-word transforms) lives in
solution.cpp. It may move into lib/easy as a multiply for a run-time NTT prime.

## Evidence that the route is tight

- The forum on the problem page has threads titled "双 log 做法卡常建议" (constant-tuning advice
  for the two-log method) and "分治FFT卡常建议" (constant-tuning advice for divide-and-conquer FFT).
  Titles from the coordinator. Not re-read: Luogu's forum needs a login.
- Our baseline (the same divide and conquer, textbook NTT) takes 680 ms on n = 262143, over the
  500 ms limit on lc-amd. Judge machines may be slower.

## Measurements

2026-10-10, lc-amd (EPYC 7B13), `gcc:13` image, `-std=c++20 -O2` (the Luogu entry in
showcase/check.py). Script: `~/explore/showcase-p5408/bench.py` (interleaved, 15-21 runs). The VM
was loaded (load average 4.5-5.6, other agents), so wall times include waits for the bench lock.
CPU time (user + sys, median, 10 ms resolution) and min wall are the numbers to read.

| n | ours, cpu med | ours, wall min | baseline, cpu med | baseline, wall min |
|---|---|---|---|---|
| 262143 | 60 ms | 59 ms | 680 ms | 680 ms |
| 131073 | 30 ms | 29 ms | 330 ms | 325 ms |
| 200000 | 40 ms | 27 ms | 350 ms | 276 ms |

Ours is 11x faster than the baseline and 8x under the limit.

- Schoolbook threshold (n = 262143, cpu median): 8: 70 ms, 16: 60 ms, 32: 60 ms, 64: 60 ms. Kept 32.
- `multimod::LazyProduct` not tried: it needs both factors within half the transform, and the
  balanced halves here have 2^k + 1 coefficients, so it would double half of the transforms.

## Checks

- showcase/check.py: sample OK, 300 stress cases against brute.cpp OK (every n <= 70, then random
  n <= 3000).
- ASan + UBSan (`-O1 -fsanitize=address,undefined`, gcc:13): n in {1, 2, 3, 31-34, 63-66, 100, 257,
  1000, 2049, 3000} match brute.cpp; n = 70000 runs clean.
- main.cpp and baseline.cpp agree on all three max cases.

## Judge caveats

- Needs C++20 (`std::bit_width`, `<bit>`): submit as C++20 or later. Not submitted.
- No -march on Luogu: the target pragma enables AVX2; the multimod kernels are inline asm.
