# CF 1770G Koxia and Bracket

https://codeforces.com/problemset/problem/1770/G. |s| <= 5·10^5, TL 5 s, 256 MB, mod 998244353.
Statement read from the Luogu mirror (https://www.luogu.com.cn/problem/CF1770G): one line s;
print the number of minimum-size deletion sets that leave s balanced.

## Problem and routes

- Greedy scan: a ')' at balance 0 is unmatched. The prefix up to the last unmatched ')' loses only
  ')'; the rest loses only '(' (reverse it, swap brackets, same count). Answer: the product.
- Over the prefix's ')': state e = deleted - unmatched so far >= 0. Matched ')': times (1 + x).
  Unmatched ')': times (1 + x^-1), negative powers dropped. Answer: [x^0] at the end.
  States above the number of unmatched ')' still to come are dropped.
- Editorial route (and ours): divide and conquer. On a segment of len ')' with c unmatched,
  states >= c never reach the floor: one product with C(len, j), shifted by -c. States < c
  recurse into the halves. A node's input has fewer than 3 len states, so O(n log^2 n).
  Leaves of <= 32 ')' are stepped directly.
- Evidence the route is tight: Codeforces blog 111050. The author's editorial-style code, KACTL's
  NTT and the editorial's own code TLE'd on test 12 with the 32-bit compiler; the same code took
  1185 ms with 64-bit. Not verified here: no 32-bit toolchain on the VM.

## Results

2026-10-10, claude. lc-amd (EPYC 7B13), gcc:14 `-std=c++23 -O2` (check.py's Codeforces
build), 5 runs each, inside the bench lock. Raw: lc-opt-explore/showcase-1770g/lc-amd/.
main.cpp is 41776 bytes (limit 64 KB), on `lib/easy/multiply.hpp`.

| Max case (n = 5·10^5) | main.cpp | baseline.cpp |
|---|---:|---:|
| 0 `())` repeated | 91 ms (111 max) | 677 ms |
| 1 `(((` + `)))))))))` repeated | 102 | 895 |
| 2 random, P('(') = 0.27 | 110 (125) | 883 |
| 3 random, P('(') = 1/3 | 96 | 675 |
| 4 random blocks `(`^a `)`^(a+b) | 82 | 658 |
| 5 unmatched on both sides | 86 | 608 |
| 6 n/4 matched, then unmatched | 84 | 748 |

- Windows build (mingw-w64 `-O2 -static`) under Wine with a persistent wineserver, 3 runs:
  main.exe 137-222 ms, baseline.exe 0.80-1.39 s on cases 0-2 (wine.txt). Without `-p`,
  wineserver startup adds 0.7-5 s per run; ignore such times.
- Shape sweep (sweep.txt, 24 shapes): baseline peaks near P('(') = 0.26-0.28 and on short
  periodic blocks; ours stays 60-180 ms (/usr/bin/time, 10 ms resolution).
- Ratio 7-9x. The baseline still passes 5 s on 64-bit; only the 32-bit judge build makes this
  route fail (blog above). The showcase is the margin, not a TLE turned AC.
- Checks: samples (Linux and Wine); 1000 small cases vs brute.cpp (all subsets, |s| <= 18) with
  leaf sizes 1, 4, ASan/UBSan and baseline; 300 medium cases (|s| <= 20000) vs an O(n^2) state DP
  (explore dp.cpp) with leaf size 1, main and baseline; ASan on 30 of them. All OK.
- baseline.cpp: textbook NTT (iterative radix-2, bit reversal, `% P`, roots per level), same
  schoolbook cutoff (factor <= 32) as easy::multiply.
