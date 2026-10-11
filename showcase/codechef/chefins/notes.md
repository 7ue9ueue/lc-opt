# CodeChef CHEFINS: Chef and Numbers

- Problem: https://www.codechef.com/problems/CHEFINS (COOK77, Dec 2016). TL 2 s.
- N, K, Q, X <= 2e5; K distinct allowed numbers in [1, N]; per query, is X a sum of allowed
  numbers, each used any number of times? Output `Yes` / `No` per line.
- Editorial: https://discuss.codechef.com/t/chefins-editorial/13686

## Intended solution

Grow the range in stages: at stage i, extend the 0/1 reachability vector to [0, 2^i), add the
allowed numbers in it, square twice. Total O(M log M), M = 2e5.

## Our route

`a` = 0/1 reachability vector over [0, max X], a_0 = 1, a_f = 1. Square with
`easy::multiply(a, a)`, truncate to max X + 1 terms, clamp to 0/1. After r squarings `a` covers
sums of up to 2^r addends; r = bit_width(max X) = 18 for X = 2e5. Each product has 400001
terms, so 18 transforms pairs of length 2^19: O(M log^2 M).

No false zeros mod P: `a` is 0/1 with at most 200001 terms, so each coefficient of a^2 counts
at most 200001 < P pairs.

Evidence the route normally fails: the editorial calls this first solution O(N log^2 N), "quite
good but will not pass the time limit". A comment by likecs on that thread says it passes with a
fast FFT, and that most naive FFT implementations will not.

## Measurements

lc-amd (EPYC 7B13), gcc:13 `-std=c++20 -O2`, 2026-10-10, worktree base 00d2abe. Interleaved,
7 runs each, one flock held for all runs (`~/explore/showcase-chefins/time.sh`), wall ms sorted:

| Input | solution (lib/easy) | baseline (textbook NTT) |
|---|---|---|
| max0: all of 1..2e5 allowed, Q = 2e5 | 66 68 69 71 75 97 103 | 758 762 766 808 833 864 1369 |
| max1: 50000 random even numbers, Q = 2e5 | 64 64 64 65 67 67 99 | 755 774 788 789 865 888 901 |

Median 69 ms against 766-788 ms: about 11x. The time does not depend on the input's shape
(always 18 squarings at 2^19).

`check.py` on lc-amd: sample OK, stress 500 cases against brute.cpp OK. Its timing step waited on
other agents' locks (medians 1089-2353 ms are lock waits), so the table above is the measurement.
ASan/UBSan (gcc:13, -O1): 61 small cases match brute.cpp; max1 output matches.
Max-case outputs of solution and baseline match (md5).

## Caveats

- The baseline also fits 2 s on lc-amd (0.77 s, 2.6x margin). CodeChef's 2016 judge was likely
  slower (guess); the editorial's claim was about that machine and complex-double FFTs.
- An O(M K / 64) bitset DP is also a known alternative (thread comments); not tried.
- gcc:13 prints `-Winvalid-constexpr` warnings from lib/poly (`ntt::detail::table_words`); not errors.
