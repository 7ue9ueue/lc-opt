# mul_mod2n_convolution

c_k = sum over i j = k (mod 2^N) of a_i b_j mod 998244353, N <= 20. 5 s.

Best judged: ours, 17 ms: [409607](https://judge.yosupo.jp/submission/409607) (`main.cpp` of #295).
Earlier: 21 ms, [409247](https://judge.yosupo.jp/submission/409247) (#115); 22 ms,
[409243](https://judge.yosupo.jp/submission/409243) (#108).
Record when opened (issue #30): 81 ms.

## Design

- Index i = 2^s u, u odd: level s. Units mod 2^M are +-u, u = 4i + 1 (i < L = 2^(M-2)); the
  positive ones form G = <5>, cyclic of order L. A product of levels s, t lands in level s + t as a
  product of units mod 2^m, m = N - s - t; s + t >= N lands on 0. Per level, the sign transform
  (x(u) +- x(-u)) leaves two functions on G.
- Round 3: transforms over G run in natural order of i (no discrete-log permutation). Stage H
  (blocks of 2H positions) maps (x_i, x_i+H) to (x_i + x_i+H, (x_i - x_i+H) t_H(i)),
  t_H(i) = r_2H^(dlog_5(4i + 1) mod 2H): a decimation in frequency twisted by a character, so
  every block stays a function on a quotient of G. Twiddles depend on H and i only, never on M.
  Blocks of 8 positions (u mod 32) are leaves: cyclic convolutions of length 8 in log order
  (one `vpermd` from natural order). Check of the math: `twisted_check.py` (exploration folder).
- Folding a factor mod 2^m keeps the "sum" halves of the top stages, so the first 2^(m-2) words of
  a transform of length L are the transform of the fold. Every level of a and b is transformed
  once (2^N words per factor); output level m sums the leaf products of its N - m + 1 pairs and
  runs one inverse.
- Radix-4 steps on blocks of 4H positions take tau_i = t_2H(i), i < H: t_2H(i + H) = J tau_i
  (J = r_4^-1, as 1 + 4H = 5^(3H) mod 16H) and t_H(i) = tau_i^2. So a step is a radix-4
  butterfly with per-position twiddles tau, tau^2, tau^3 and the constant J (4 Shoup products).
  Tables: per vector of positions, lines [q, w] of tau, tau^2, tau^3 for quarters h = 1, 4, 16, ...
  vectors, plus (N even) the radix-2 top of level 0, t_H(i) for i < H/2. 1.5 MB for N = 20; the
  inverse tables replace the forward ones after the products.
- Table values: chi(u) = r^dlog(u) for u < 2^(N-2) as chi(u_lo) theta^(u_hi u_lo^-1) with
  u = u_lo + 2^b u_hi, 2b >= N (1 + 2^b z = (1 + 2^b)^z there): eight u_lo are a geometric
  sequence in u_hi. Lower tables are fourth powers.
- Input: each 2^16-value chunk is deinterleaved level by level: level s takes positions
  2^s (4i + 1) (P) and 2^s (4i + 3) (R, stored reversed, so R[i] = x(-(4i + 1))); evens go on to
  level s + 1. Levels past the chunk's vector reach (and small N) come from a short "rest" array.
- Forward: the sign transform is a pass of its own (P + R, P - R), then the top step per sign,
  then depth-first radix-4 blocks (one sign per step; in tiles of 256 vectors both signs).
  Steps on both signs at once, or the top step fused with the sign transform, read and write
  eight or more streams and lose on large levels (below).
- Leaf v lies in levels 0..k (k from v's band). For each output level t <= k, the sum over s of
  a_s b_(t-s) at leaf v is stored over b_t (b_t is last read by level t). Windows [x, x] of a's
  leaves on the stack (z^j x is an unaligned load), 64-bit `vpmuludq` sums, only the high dword
  reduced between products, one Montgomery reduction per output vector. Two leaves and two levels
  (t, t - 1) at a time: eight independent chains (one is latency-bound); band 0 (one product per
  leaf) four leaves at a time. An `asm` barrier keeps GCC from spilling the sums.
- Inverse: blocks, top step per sign, then a pass for the sign combination and the scale
  (nv^-1 2^32 / 2: transform, Montgomery factor, sign transform).
- Folds to 32 words (output levels m <= 7, sums) from the first four leaves: one inverse radix-4
  step, times 1/4, then to log order. Output levels m <= 7: direct cyclic convolutions.
- Output: levels interleaved back by chunks (the inverse of the input deinterleave), formatted by
  `../convolution_mod/fields.hpp` (fixed-width fields). `.preinit_array` start, `_exit`.
  The input mapping is advised `MADV_SEQUENTIAL` (`io::advise_sequential`).

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
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  20.57 → 20.24 ms (0.986). 47/47 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).

- 2026-10-10, claude (round 3). Exploration files (scripts, variants, benchmarks):
  `~/Documents/cpp_hpc/lc-opt-explore/mul_mod2n_convolution/` (Mac) and its `lc-*` copies.
  Phases: in-process `CLOCK_MONOTONIC` stamps, n_equals_20, `lc-bench`, median of 15-21 runs.
  - Natural-order twisted transform (Design) instead of the discrete-log permutation (grid
    transposes, gathers, scatter: 3.8 ms in round 2). First cut, intrinsics: `judge.py bench`,
    `lc-bench`, 11 rounds: 22.45 -> 19.99 ms (0.892). Phases: split 1.72 + 1.73 (parse and
    deinterleave), forward 1.76 + 1.74, products 1.84, inverse 1.98, output 4.48.
  - Output phase: `write(2)` 3.65 ms, format 0.67, interleave 0.17 (variants without them).
    Deinterleave plus first touch of the level arrays: 0.27 ms per factor.
  - GCC did not inline the butterflies and copied twiddles through the stack: forced inlining and
    twiddles loaded straight from table lines: forward 1.76 -> 1.57, inverse 1.98 -> 1.79.
  - Transform microbenchmark (`bench_kernel.cpp`, ns per vector-stage, both signs): in L1
    (2^8 vectors) forward 0.69, inverse 0.83; lib/ntt's transform (round 2 code) 0.63 / 0.64.
  - Steps on one sign at a time instead of both: 2^15 vectors forward 0.875 -> 0.747, inverse
    1.00 -> 0.85. Sign transform and sign combination as passes of their own: top step at 2^14
    126 -> 72 us; forward at 2^14 0.857 -> 0.736, inverse 0.915 -> 0.764. Phases: forward 1.47 ->
    1.39, inverse 1.85 -> 1.63 ms. Eight or more streams per loop are the cost, not the passes.
  - Leaf products (`bench_products.cpp`): GCC kept the 64-bit sums on the stack between products;
    an `asm` barrier: 1.86 -> 1.65 ms; two levels per window: 1.50; two leaves at a time: 1.34;
    band 0 four leaves, no final reduction: 1.31. Per product 3.9 ns in deep bands (13.5 cycles),
    6.6 in band 0, 5.4 in band 1; the data being in L3 or L1 makes no difference.
  - Tables: one direction at a time (inverse rebuilt over the forward ones after the products),
    chains of a row interleaved, theta powers from a table: 0.47 -> 0.26 ms (1.5 MB instead of 3).
  - No gain, dropped: fused radix-8 top (radix-2 step and both halves' radix-4, both signs):
    forward 1.57 -> 2.27 ms per factor; four butterflies per iteration (two j, two signs), or two
    j per iteration: same in L1, worse at 2^15; `#pragma GCC optimize("schedule-insns")`: same;
    software prefetch in top steps (16, 64 vectors ahead) or in the products: same or +2%; big
    steps (h >= 256 or 1024) as two radix-2 passes: worse (2^15: 0.73 -> 0.84); odd lanes loaded 4
    bytes on in the inverse butterflies: 0.84 -> 0.86; four sums per product: 1.85 -> 1.96 ms;
    `vbroadcastss` for leaf coefficients: same; windows 2-4 leaves ahead: band 0 9.4 -> 7.9
    ns/product, superseded; tiles of 64-1024 vectors: same.
  - Zen 3 (`uops.cpp`, `lc-amd`, cycles per instruction, independent): `vpmulld` 0.50, `vpmuludq`
    0.61, both mixed 0.62, `vpaddd` 0.42, `vpminud` 0.55, `vpblendd` 0.58.
  - Final phases: tables 0.26, split 1.63 + 1.58, forward 1.39 + 1.40, products 1.33, inverse
    1.60, output 4.41; in-process 13.7 ms (15.8 for the first cut).
  - `judge.py bench`, slowest 3 cases: `lc-bench` 21 rounds, base 19.97 -> 16.65 ms (0.844);
    `lc-intel` 11 rounds, base 17.32 -> 16.00 ms (0.926).
  - Checks: 47/47 official tests (`lc-amd`, slowest 17.1 ms); `stress.py` 300 rounds plus 10 known
    N = 20 cases (in the judge image); random N = 0..14 against `brute.cpp`; ASan/UBSan on all 47
    official cases, file and pipe input.
  - CI (#295, three EPYC 7763 runners, 21 rounds): ratios 0.852, 0.819, 0.850.
  - Submitted the merged `main.cpp` (#295): [409607](https://judge.yosupo.jp/submission/409607)
    AC 17 ms, 22.9 MiB, and [409608](https://judge.yosupo.jp/submission/409608) AC 17 ms; both
    clean 17 ms (`tools/spikes.py`), slowest case large_01 (n_equals_20, large_00: 16 ms).
- Next: transforms run 0.69 ns per vector-stage in L1 against 0.63 for lib/ntt's scheduled asm
  (a problem-local generator from `lib/ntt/gen_kernels.py` could close it); levels with an odd
  stage count pay one more pass (radix-2 top); products sit at 13.5 cycles each against about 10
  for the `vpmuludq` bound; a fused recursion over all levels (products and inverse steps per
  block, twiddles shared) is untried.
- 2026-10-10, claude (lib, issue #156 round 3): `advise_sequential` comes from
  `lib/io/sequential.hpp` (`io::advise_sequential`) instead of a local copy. Same stripped
  executable as before (judge flags, `lc-amd`).

## Sources

- Structure of (Z/2^N)^*: {+-1} x <5> (standard number theory). No code read.
- `lib/ntt` for Shoup arithmetic and roots (round 3: no transform kernels), `lib/io`,
  `../convolution_mod/fields.hpp`.
- Round 3: the character-twisted decimation in frequency over <5> in natural order was derived
  here (twisting is the standard idea of e.g. D. J. Bernstein, "The tangent FFT", 2007); no code
  read. `MADV_SEQUENTIAL` on the input: `../gcd_convolution/solution.cpp`.
