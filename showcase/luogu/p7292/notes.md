# Luogu P7292 Chasse Neige 加强版

https://www.luogu.com.cn/problem/P7292 (original: P7289, r <= 2e5, 400 ms).

- Task: T queries (n, k): permutations of length n with p1 < p2, p(n-1) > p(n) and exactly k peaks
  (2 <= i <= n-1, p(i-1) < p(i) > p(i+1)), mod 998244353.
- Limits: T <= 2e5, 3 <= n <= r <= 1e6, max(1, floor((n-1)/2) - 10) <= k <= floor((n-1)/2).
  800 ms, 256 MiB. Input `T r`, then T lines `n k`.
- Sample: `2 10 / 3 1 / 5 2` gives `2 16`.
- The statement says r = 1e6 should stop the O(n log^2 n) divide and conquer FFT.

## Intended solution

From alpha1022's write-up (https://alpha1022.me/articles/lg-7289-7292.htm) and the P7289 题解
https://www.luogu.com.cn/article/xuh3dny8:

- g(n, i) = i g(n-1, i) + (n - i) g(n-1, i-2) + 2 g(n-1, i-1) (insert the maximum; g(n, 2k) is the
  answer, g(n, 2k+1) the same count ending up). With h(n, j) = g(n, n - j): h(n, 1) = E_n, the
  zigzag numbers, EGF tan + sec; h(., 0) = 0.
- Solved for h(n-1, j+1), the recurrence fills rows j = 2..22 from row 1 in O(22 r). The answer is
  h(n, n - 2k), and n - 2k <= 22 by the limits on k.
- tan + sec = (1 + sin) / cos: one series inverse, O(r log r).

## Our route

E_n from the ODEs both write-ups derive, F' = F^2 + 1 (tan), G' = F G (sec), as ordinary series:
(n + 1) a_(n+1) = [x^n] a^2 + [n = 0], (n + 1) b_(n+1) = [x^n] a b. Two online convolutions by
plain binary CDQ (the abc222_h_dc pattern): per node, a[l, m) a[0, L) (twice), a[l, m) b[0, L) and
b[l, m) a[0, L) with `easy::multiply` (three products; two at l = 0), schoolbook leaves of 32.
O(r log^2 r), size 2^20. Then the row recurrence, two rows kept, queries answered by row.

## Results

2026-10-10, claude. Plain products (step 1). lc-bench (EPYC 7B13, load 0.00), gcc:13
`-std=c++20 -O2` (check.py's Luogu entry), `showcase/bench.py --rounds 3`, median.

| Input | solution.cpp | baseline.cpp |
|---|---:|---:|
| max0: r = 1e6, T = 2e5, random n | 531 ms (66% of TL) | 5686 ms |
| max1: r = 1e6, T = 2e5, n in {r - 1, r} | 529 ms | 5683 ms |
| r = 2e5, T = 2e5 (P7289 size, 400 ms TL) | 127 ms (125-130) | 1013 ms (1010-1019) |

- Under 90% of the TL, so step 2 (middle products, `easy::Cyclic`) was not built.
- main.cpp: 39795 bytes (lib/easy/multiply.hpp, compacted).
- Peak RSS on max0: 80752 KB (`/usr/bin/time -f %M`, lc-amd).
- lc-amd, check.py: 528 / 524 ms.

## Checks

- brute.cpp: enumeration for n <= 8, the g recurrence above. The recurrence matched enumeration
  for all n <= 10 and all k (one-off, `lc-opt-explore/showcase-p7292/dpcheck.cpp`).
- check.py on lc-amd: sample OK, 300 stress cases OK (every r in 3..42, then r <= 2000).
- ASan + UBSan (`-O1`, gcc:13): 13 small seeds match brute.cpp; r = 70000 clean.
- main and baseline agree on both max cases and on r = 2e5.

## Judge caveats

- Not submitted. Needs C++20. No -march on Luogu: the target pragma in lib/easy gives AVX2.
- 66% of the TL on lc-bench; the judge's CPU may be slower (the brief asks for 2x margin).
