# ABC222 H Beautiful Binary Tree: divide and conquer route

https://atcoder.jp/contests/abc222/tasks/abc222_h. N <= 10^7, TL 3 s, 1024 MiB, mod 998244353.
The second solution for this problem; the first (pow, O(N log N)) is in `../abc222_h`.

## Problem and routes

- A(x) = x (1 + 3A + A^2)^2, answer a_N (editorial, https://atcoder.jp/contests/abc222/editorial/2742).
- Intended: O(N) from a P-recursive recurrence. The editorial says the O(N log N) series route
  does not make the TL.
- Ours: S = A^2, D = 1 + 3A + S, T = D^2, a_n = t_(n-1). S and T are online self-convolutions.
  Plain binary CDQ: `solve(l, r)` recurses left, adds f[0, m)^2 (l = 0) or 2 f[l, m) f[0, r - l)
  (l > 0) into [m, r) with `easy::multiply`, recurses right. Schoolbook leaves of 32. O(N log^2 N),
  products up to 2^24. I/O with std::cin/std::cout.
- baseline.cpp: the same code with a textbook NTT (iterative radix-2, bit reversal, `% P`).
  It splits products longer than 2^23 (P - 1 = 119 2^23), needed only at the root for N > 2^23.

## Results

2026-10-10, claude. lc-bench (EPYC 7B13), gcc:15.2.0 `-std=gnu++23 -O2 -march=native`,
`showcase/bench.py --rounds 3` (median; main and baseline alternated under the bench lock).

| N | solution.cpp | baseline.cpp |
|---|---:|---:|
| 10^7 | 4674 ms (156% of TL) | 65460 ms |
| 8388609 | 4131 ms | 58883 ms |
| 10^6 | 319 ms (3 runs: 317-322) | 3991 ms (3989-4012) |

- Does not fit 3 s at N = 10^7. Accepted by the user: this is the natural algorithm.
- Samples, 300 random N (check.py stress) and both max cases against brute.cpp: OK.
  ASan/UBSan build on N in {1, 2, 3, 33, 64, 65, 222, 3000, 222222}: clean.

## Tuned variants (exploration only, not shipped)

Files and scripts in `~/Documents/cpp_hpc/lc-opt-explore/showcase-abc222_h_dc/`. N = 10^7,
lc-bench, median of 5, all checked against brute.cpp.

| Variant | Time |
|---|---:|
| v1: B-ary relaxed (B children per node, block transforms reused, `poly::Transform::multiply_add`), B = 16 | ~3.7 s (lc-amd, loaded; 68% in leaf products) |
| v2: + leaves split to pointwise values (8 x 8 leaf products -> 1 product per value), twist tables | B = 16: 1734 ms; B = 8: 1941; B = 4: 2010; B = 32: 2034 |
| v3: + twist on the fly, l = 0 nodes split too, 32-term vector leaf dot | B = 8: 1269; 16: 1328; 32: 1374; 4: 1497; 2: 1932 |
| v4: + lazy reductions in the split, vector accumulate | B = 8: 1148; 16: 1229 |
| v5: + leaf pushes each new value to later targets (no store-forward stalls) | B = 8: 1052; 16: 1133 |

- v5 at B = 8: 1052 ms, 35% of the TL. Profile (v4, B = 8): split 30%, lib transforms 30%,
  pointwise sums 20%, leaf 10%.
- Splitting needs roots of x^8 - w_p: w_p from lib/poly's leaf order; transforms up to 2^23
  only, so the root node may have more than B children.
