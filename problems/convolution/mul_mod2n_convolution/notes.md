# mul_mod2n_convolution

c_k = sum over i j = k (mod 2^N) of a_i b_j mod 998244353, N <= 20. 5 s.

Best judged: ours, 21 ms: [409247](https://judge.yosupo.jp/submission/409247) (current `main.cpp`, #115).
Earlier: 22 ms, [409243](https://judge.yosupo.jp/submission/409243) (#108).
Record when opened (issue #30): 81 ms.

## Design

- Index i = 2^s u, u odd: level s. Units mod 2^M are +-5^k, k < 2^(M-2). A product of levels s, t
  lands in level s + t as a product of units mod 2^m, m = N - s - t; s + t >= N lands on 0.
  Per level, the sign transform (x(u) +- x(-u)) leaves two cyclic sequences of length 2^(M-2).
- Folding a factor mod 2^m is folding its sequence to length 2^(m-2). In `lib/ntt`'s transform
  tree (leaves x^8 - w, group k's children 4k + t, radix-2 top for odd lengths) the first 2^(m-2)
  words of a transform of length 2^(M-2) are the transform of the folded sequence. So every level
  of a and b is transformed once (2^N words per factor), and output level m sums the leaf products
  of its N - m + 1 pairs and runs one inverse.
- Leaf v lies in levels 0..k (k from v's band). One pass over v computes, for every output level
  t <= k, the sum over s of a_s b_(t-s) at leaf v and stores it over b_t (b_t is last read by
  level t). No accumulator buffer; the inverse runs in place in b's parts.
- Leaf product mod x^8 - w: a window [w x, x] on the stack, so x^j x mod x^8 - w is an unaligned
  load; 64-bit sums of eight `vpmuludq` products. The windows of leaf v + 1 are built before the
  products of leaf v: loads from fresh stores stall. Between products only the high dword is
  reduced (min with high - 2P); one Montgomery reduction per output vector.
  The scale (nv^-1 2^32 / 2) undoes the transform, the Montgomery factor and the sign transform.
- Folds to 32 words (for output levels m <= 7 and the sums) come from the first four leaves of
  each transform: one inverse group, times 1/4.
- The h = 1 group and the leaves are done here (the lib's bottom kernels fuse forward, product
  and inverse); the other groups use the lib's kernels.
- Output levels m <= 7: direct cyclic convolutions of 32-word folds.
- Permutation (levels with M >= 13): x is transposed once into a grid, grid[c][h] = x[c + 2^B h],
  B = N - 8, rows 272 words apart (256 + 16, so rows 2^s apart start in different L1 sets).
  Level s, b = M - 8, has y[u] = x[2^s u] in row 2^s (u mod 2^b), column u >> b. The row of +-5^k
  depends only on k mod 2^(b-2), so eight consecutive k_lo use 16 rows (17 KiB) for all k_hi:
  `vpgatherdd` from L1. The gather also does the first layer of the forward transform. The input is
  parsed 16 rows of x at a time into a 256 KiB buffer and transposed by 8x8 blocks. The output
  scatters every level into the grid, then transposes it back 16 rows at a time and formats them.
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
  - Submitted the merged `main.cpp` (#108): [409243](https://judge.yosupo.jp/submission/409243)
    AC 22 ms, 32.9 MiB.
- Next: leaf products (about 10% in `perf`; SoA layout, or scalar leaves with pointwise
  products), page faults (`perf` 8% kernel; 21 MB touched), `vpgatherdd` vs scalar loads in the
  permutation, scatter_units' scalar stores.
- 2026-10-09, claude (round 2). `lc-amd`, judge flags; phases are in-process `CLOCK_MONOTONIC`
  stamps on n_equals_20, both factors together.
  - Floor (`floor.py`, 11 rounds, 3 largest cases): 13.19 ms with `write_array`, 12.06 ms with
    fixed-width fields. Base (round 1) in the same setup: 21.3-22.5 ms.
  - Base phases (ms): split 6.7 (parse 3.05 of it), forward 2.7, leaf products 1.8, inverse 1.1,
    output 2.15. `perf` on `lc-intel`: kernel 12%, leaf products 16%, gather 10%, transposes 17%.
  - v3: leaf products per leaf over all levels, in place over b; windows on the stack (no [w x, x]
    copy of a, no accumulator: 6 MB less). Forward 2.7 -> 2.33, leaf products 1.8 -> 2.25.
    `perf stat` on `lc-intel`: 0.91 M `ld_blocks.store_forward`.
  - v4: windows of leaf v + 1 before the products of v: 0.30 M blocks, leaf 2.25 -> 2.03.
    Windows two leaves ahead: 0.05 M blocks and -4.6% cycles on `lc-intel`, but leaf 2.03 -> 2.10
    on `lc-amd`; dropped.
  - v5: forward top layer fused into the gather; folds from the first four leaves (no fold pass).
    split + forward 9.0 -> 8.3, total in-process time unchanged within noise.
  - Scalar loads instead of `vpgatherdd` (`_mm256_setr_epi32` of eight loads): split 3.05 vs 3.07;
    no gain, dropped.
  - v7: high-dword reduction between products, one Montgomery per output: no change (2.03 vs
    2.04). Kept: fewer multiplies. Without any products the leaf phase takes 0.70 ms, so the
    products cost about 15 cycles each, near one `vpmuludq` per cycle (16 per product). Unrolling
    the 8-step product loop (GCC kept it rolled): no change.
  - v8: one grid for all levels instead of rows + evens per level (above). Split 6.2 -> 5.75.
    Phases now: parse 3.0, transpose in 0.78, gathers 1.6 (level 0 0.77, level 1 0.69),
    forward 2.1, leaf 2.0, inverse 1.1, scatter 0.77, transpose out 0.62, format 0.66.
  - v10: sign combine fused into the scatter. Neutral (leaf + inverse + output 5.32 vs 5.33 ms);
    kept, one pass less.
  - Page zeroing (`lc-amd`, a C probe): 0.05 ms per MB with `MADV_HUGEPAGE`, 0.42 without.
  - `tools/judge.py bench`, 21 rounds, slowest 3 cases: `lc-amd` base 22.51, v8 21.21, v10 20.76 ms
    (ratio 0.924); `lc-intel` base 20.97, v10 17.84 ms (ratio 0.860).
  - Checks: 47/47 official tests (slowest 20.2 ms); `stress.py` 300 rounds plus 10 N = 20 known
    cases; random N = 13-16 against `brute.cpp` (7 cases); ASan/UBSan on all 47 official cases
    and pipe input.
  - Submitted the merged `main.cpp` (#115): [409247](https://judge.yosupo.jp/submission/409247)
    AC 21 ms, 24.9 MiB (was 22 ms, 32.9 MiB).
- Next: the products are `vpmuludq`-bound (about 1.3 of the 2.0 ms): a Karatsuba short product
  over levels saves about 12% of them (guess). Gathers run at about 2.2 cycles per element on
  level 0 and 4 on level 1. Forward and inverse are the lib's kernels (3.2 ms).

## Sources

- Structure of (Z/2^N)^*: {+-1} x <5> (standard number theory). No code read.
- `lib/ntt` for transform kernels and tables, `lib/io`, `../convolution_mod/fields.hpp`.
