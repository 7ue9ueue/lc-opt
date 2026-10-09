# mul_mod2n_convolution

c_k = sum over i j = k (mod 2^N) of a_i b_j mod 998244353, N <= 20. 5 s.

Best judged: none yet. Record when opened (issue #30): 81 ms.

## Design

- Index i = 2^s u, u odd: level s. Units mod 2^M are +-5^k, k < 2^(M-2). A product of levels s, t
  lands in level s + t as a product of units mod 2^m, m = N - s - t; s + t >= N lands on 0.
  Per level, the sign transform (x(u) +- x(-u)) leaves two cyclic sequences of length 2^(M-2).
- Folding a factor mod 2^m is folding its sequence to length 2^(m-2). In `lib/ntt`'s transform
  tree (leaves x^8 - w, group k's children 4k + t, radix-2 top for odd lengths) the first 2^(m-2)
  words of a transform of length 2^(M-2) are the transform of the folded sequence. So every level
  of a and b is transformed once (2^N words per factor), and output level m sums the leaf products
  of its N - m + 1 pairs and runs one inverse.
- Leaf product mod x^8 - w: a's leaves are stored as [w x, x], so x^j x mod x^8 - w is an
  unaligned load; 64-bit sums of eight `vpmuludq` products, one Montgomery reduction per pair.
  The scale (nv^-1 2^32 / 2) undoes the transform, the Montgomery factor and the sign transform.
- The h = 1 group and the leaves are done here (the lib's bottom kernels fuse forward, product
  and inverse); the other groups use the lib's kernels.
- Output levels m <= 7: direct cyclic convolutions of 32-word folds.
- Permutation (levels with M >= 13): the odd values as rows[t][h] = x[2t + 1 + 2^b h], b = M - 8.
  The row of +-5^k depends only on k mod 2^(b-2), so eight consecutive k_lo use 16 rows (16 KiB) for
  all k_hi: `vpgatherdd` from L1. x -> rows by 8x8 transposes, 16 rows of x per pass; the input of
  level 0 is parsed 16 rows at a time into a 256 KiB buffer. The output runs the same steps
  backwards; level 0 is formatted 16 rows at a time.
- Output: `../convolution_mod/fields.hpp` (fixed-width fields). `.preinit_array` start, `_exit`.

## Log

- 2026-10-09, claude (round 1). `lc-amd`, judge flags.
  - v1: plain discrete-log gather (`x[5^k << s]`) and scatter into c, leaves as above.
    47/47 official tests, slowest 25.3 ms (n_equals_20). Phases (ms, n_equals_20): read 4.0,
    split 6.5 (of which 2.2 a fold reading each part 32 times; fixed), forward 3.3, levels 4.5,
    write 4.3. `perf` on `lc-intel`: split 21%, scatter 15%.
  - v2: row permutation (above), a's parts in the upper halves of their [w x, x] buffers, input
    and output of level 0 in bands. Phases: per factor parse + to_rows 2.2-2.7, level-0 gather
    0.45-0.6, other levels 0.7-0.85; forward 2.7; levels + write 8.5.
    Fusing the parse with to_rows saved nothing measurable (read + split 7.45 -> 7.5 ms).
  - `tools/judge.py bench`, 11 rounds, slowest 3 cases: v1 26.55, v2 22.41 ms (ratio 0.849).
  - Checks: 47/47 official tests (slowest 21.3 ms); `stress.py` 300 rounds N <= 14 plus 10 N = 20
    cases with known answers (a delta factor); ASan/UBSan on all 47 official cases and pipe input.
  - Floor estimate (guess from phases): parse 4.0 + format and write 4.3 + start and exit ~1.5.
- Next: leaf products (about 10% in `perf`; SoA layout, or scalar leaves with pointwise
  products), page faults (`perf` 8% kernel; 21 MB touched), `vpgatherdd` vs scalar loads in the
  permutation, scatter_units' scalar stores.

## Sources

- Structure of (Z/2^N)^*: {+-1} x <5> (standard number theory). No code read.
- `lib/ntt` for transform kernels and tables, `lib/io`, `../convolution_mod/fields.hpp`.
