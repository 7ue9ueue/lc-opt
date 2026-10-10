# multipoint_evaluation_on_geometric_sequence

f of N coefficients, M, a, r mod 998244353, N, M <= 2^19; print f(a r^i) for i < M. 10 s.
Slowest tests: max_random_00..02 (N = M = 2^19, random a, r). The a0_r01 tests have a = 0 or
r in {0, 1}: no product (2.5-4.7 ms). near_pow_of_2 has N, M near 2^18; nm_1 has N or M = 1.

Best judged: ours, 10 ms: [409475](https://judge.yosupo.jp/submission/409475) (`main.cpp` of #227),
clean (`tools/spikes.py`).
Record when opened (issue #76): 34 ms.

## Design

- Chirp z-transform (Bluestein): with t(k) = k (k - 1) / 2 and any g,
  i j = t(i + j - g) - t(i) - t(j - g) + g i, so f(a r^i) = r^(g i - t(i)) sum_j A_j K_(i + j)
  with A_j = c_j a^j r^-t(j - g) and K_u = r^t(u - g). A middle product: one cyclic convolution of
  length L >= N + M - 1 of A (at [0, N)) and the reversed kernel B_v = K_(L - 1 - v) (all of
  [0, L)) gives f(a r^i) = r^(g i - t(i)) (A B)[L - 1 - i], so the outputs sit at the top end in
  reverse order, aligned to vectors for any M.
- g = L / 2 - 1 (round 2): t(1 - x) = t(x) makes K, and so B, a palindrome, B_(L-1-v) = B_v.
  Then b mod x^s - 1/z = z rev(b mod x^s - z) for every node of the transform (rev: the s
  coefficients reversed), and with the twiddle table's bit-reversed order the conjugate of node k
  is 3 * 2^e - 1 - k (2^e <= k < 2^(e+1)). b's forward levels run for one node of each pair; the
  other's groups (s = 32, the level above the bottom stage, which only reads b) are the first's
  in reverse order, each times its z (`mirror`). At the top: blocks 2-3, 4-7, 5-6 pair up;
  blocks 0 and 1 (z = 1, -1) recurse the same way. B is generated for v < L / 2 only, and its
  radix-8 pass reads that half (column j with column q - 1 - j) and writes blocks 0, 1, 2, 4, 5.
- Chirps: `lib/poly/chirp.hpp`: x_k = c s^k q^t(k), 32 per step, x_(k+32) = x_k g_k with
  g_(k+32) = g_k q^1024, one Montgomery and one Shoup product per vector. With t(u - g) =
  t(u) + t(-g) - g u: K_u = r^t(-g) (r^-g)^u r^t(u), A_k = c_k r^-t(-g) (a r^g)^k r^-t(k).
- The convolution is convolution_mod's (`ntt::detail::forward_radix8`, the bottom kernels of
  `lib/ntt/product_kernels.hpp`, a local copy of `ntt::detail::Subtrees` with the conjugate
  pairs and an option to skip b's forward levels) with two more changes: B's radix-8 pass above;
  and the last radix-2 level computes only the upper half, (u - w) [L/2 - 1 - i], times
  r^(g i - t(i)) and the transform's scale, reversed.
- L = 2^lg >= 2 max(N, M) with lg even (convolution_mod's 2 * 4^j vector layout), so A and the
  outputs each lie in one half. Odd lg rounds up: near_pow_of_2 (N, M ~ 2^18) uses 2^20, not
  2^19 (7.3 ms against 9.2 for max_random; the score is unaffected).
- a = 0 or N = 1: c_0 everywhere (only c_0 is read). r = 0, r = 1 or M = 1: f(a) by Horner's
  rule in x^32 over 4 vectors of lanes, then c_0 or f(a).
- `lib/io` input (`io::read_bulk`) into the transform buffer; output in 10-byte fixed-width
  fields (`problems/convolution/convolution_mod/fields.hpp`, judge-specific). One mapping in
  huge pages for A, B, the tables and the text. The program runs from `.preinit_array` and
  ends with `_exit` (`lib/run/early.hpp`).

## Floor

`lc-amd`, whole process, judge flags, max_random_00..02, interleaved (scratch `timeit.py`,
judge.py's build and runner without the checker): read N + 4 numbers with `io::read_bulk` and
write M values with `fields.hpp`, nothing else (`floor.cpp`): 4.92-4.99 ms median (15 rounds).

## Log

- 2026-10-10, claude (round 1): first solution. All on `lc-amd`, judge flags, max_random_00..02
  unless noted. Phases are medians of 7 runs of a probe with `CLOCK_MONOTONIC` stamps (stdout to
  /dev/null). Files: `lc-opt-explore/multipoint_evaluation_on_geometric_sequence/`.
  - v1, `ntt::Convolution` (A sparse, B full, declared as (L/2, L/2 + 1)), separate chirp
    passes: 10.69 ms (11 rounds, floor 4.92 in the same run). Phases (ms): parse 0.85,
    chirp B 0.45, weights 0.23, `multiply()` 4.70, output 0.24, format 0.45.
  - `poly::Transform` instead (forward of B, then `cyclic_product` of A with `Half::kUpper`):
    forward 1.3, `cyclic_product` 3.27: 4.57 against 4.70 for the full product. Not kept.
  - v2, convolution_mod's transform with a full radix-8 first pass for B and the upper half
    only in the last level, fused with the output's chirp: 10.15 ms, ratio 0.948 against v1
    (15 rounds). Phases: tables 0.04, first passes A 0.12 + B 0.14, halves 1.89 + 1.91, their
    top inverse groups 0.05 + 0.07, output 0.27 (0.26 when rerun on cached data:
    compute-bound), and 0.21 ms of page faults for the 5 huge pages (measured by touching them
    first).
  - Lost, v3: chirp generation, weighting and output fused into the first and last passes
    (block by block through L1, 8 KiB per 4 streams; the last inverse group by intrinsics):
    10.62 ms against 10.21 (1.037). First pass of B 0.61 (vs 0.44 + 0.16 separate), of A 0.43
    (vs 0.21 + 0.17), output 0.41 (vs 0.07 + 0.27). The passes are compute-bound (Montgomery
    chain), so fusion saves no memory time; GCC also left the 4- and 8-stream loops rolled
    with stack arrays until given `#pragma GCC unroll`.
  - Chirp cost: a step is 6.3 cycles per vector (B: 0.235 ms for 2^17 vectors), close to its
    10 multiplies on two pipes. 2^18 steps in all (B, weights, outputs): ~0.5 ms.
  - Kept (v5 = v2 with `lib/poly/chirp.hpp`, L >= 2 max(N, M)): `judge.py bench`, 21 rounds,
    slowest 3 cases: v1 10.66 -> 10.09 ms, ratio 0.9505.
  - Checks: 25/25 official tests (`judge.py test`, `lc-amd` and `lc-intel`); `stress.py` 400
    rounds (`lc-intel`; N M <= 4 10^6 against `brute.cpp`, larger by Horner's rule at 8 points;
    a, r in {0, 1, 2, P - 1} or random; sizes 1-8, near powers of two, up to 2^19); ASan/UBSan
    on all 25 official cases, file and pipe input; `lib/poly/test.cpp` at -O2 (native and
    x86-64-v3) and ASan/UBSan.
  - `lc-bench`, 21 rounds, max_random_00..02 (merged version): 10.04 ms; floor 4.70; v1 10.60.
  - Merged as #227 (new problem: CI checks only).
  - Submitted the merged `main.cpp` twice (2 of 5 this session):
    [409473](https://judge.yosupo.jp/submission/409473) AC 13 ms, from a launch spike on nm_1_01
    (13 ms; the same work in a0_r01_00 took 4; `spikes.py`: clean 10), and
    [409475](https://judge.yosupo.jp/submission/409475) AC 10 ms, clean: random_01 and
    max_random_02 10, max_random_00/01 9, the rest at most 8.
- Next: the transform is 4.1 of the 5.2 ms above the floor and is convolution_mod's. Chirp
  steps by x_(32v+l) = D_v E_l(v) (per-lane geometric E by a Shoup product, scalar D_v with its
  quotient computed on the scalar side): 8 multiplies per vector instead of 10, ~0.08 ms.
  Odd lg (radix-4 top) for mid sizes, which only helps cases below the score.
- 2026-10-10, claude (lib, issue #156 round 2): convolution_mod's `Subtrees`, `forward_radix8` and
  `bottom.hpp` moved to `lib/ntt/product.hpp`; the copies here are gone and the problem uses
  `ntt::detail::Subtrees` and `forward_radix8`. Same asm kernels; GCC now keeps `visit` and the
  radix-8 pass out of line (they were partly inlined). `judge.py bench`, `lc-bench`, 31 rounds,
  slowest 3 cases: 9.87 -> 9.82 ms, ratio 0.9965 (noise). 25/25 official tests, `stress.py` 300
  rounds, ASan/UBSan on all 25 official cases (file and pipe input).
- 2026-10-10, claude (round 2): conjugate pairs for a palindromic kernel. All on `lc-bench`,
  judge flags, max_random_00..02 unless noted. Phases: medians of 15 runs of a probe with
  `CLOCK_MONOTONIC` stamps (stdout to /dev/null), max_random_00. Files:
  `lc-opt-explore/multipoint_evaluation_on_geometric_sequence/r2/`.
  - Phases of #227's version (ms): parse 0.87, tables 0.10, chirp B 0.35, weights 0.20, radix-8
    A 0.17 and B 0.14, visits with the top inverse groups 3.99, output 0.29, write 0.34.
  - b's forward levels cost 0.86 ms: visits take 4.00 with all of them and 3.14 with none
    (timing-only builds).
  - Kept (v6): g = L / 2 - 1, B a palindrome; b's forward for one node of each conjugate pair,
    `mirror` for the other (Design). Phases: chirp B 0.22 (half the terms), radix-8 B 0.17,
    visits 3.69 (-0.30; the pairs skip 48% of b's levels, the mirror costs ~0.08: running it
    twice adds 0.07-0.09). Whole process (scratch `timeit.py`, 21 rounds): 9.79 -> 9.39 ms,
    ratio 0.9532; `judge.py bench` (11 rounds): 9.72 -> 9.41, 0.9596. Floor 4.72 in a later,
    slower run (base 10.02, v6 9.68).
  - Lost or no gain (visits unless noted, v6 in the same run in parentheses):
    - v7: the conjugate's groups mirrored into an L1 buffer per tile, no b storage for them:
      3.705, 3.712 (3.683, 3.680); the buffer 64 bytes past a 4 KiB boundary: 3.731, 3.746.
    - v8: `product_kernels::inverse_top` (both halves, canonical, scaled), then the output from
      the upper half: visits 3.56-3.59 + inverse_top 0.21 + output 0.23, against 3.69 + 0.28.
    - v9: chirp split x_(32j+l) = d_j e_l(j) in `chirp.hpp` (per-lane Shoup step of e, Shoup by
      the scalar d_j with its quotient from the scalar side: 8 multiplies per vector instead of
      10). Warm, 2^19 terms: fill 0.116 -> 0.110 ms, weights 0.194 -> 0.204, an output-like pass
      0.203 -> 0.199; in process: output -0.033, weights +0.01. Not kept (~0.03 ms in all).
    - v10: each primary tile mirrors its b while in L1: 3.704, 3.707 (3.703, 3.727).
    - v11: kernel shifted by 16 (B' = x^16 B, symmetric about 15.5 mod L), so conjugate groups
      are plain reversals without the factor z; radix-8 pass and the first 16 outputs adjusted:
      3.675, 3.678 (3.686, 3.685); v12 = v11 with v7's buffer: 3.654, 3.674 (3.700, 3.682).
      Within noise; not worth the special cases.
  - Checks: 25/25 official tests (`judge.py test`, `lc-amd`); `stress.py` 400 rounds (`lc-amd`);
    ASan/UBSan on all 25 official cases, file and pipe input (`lc-amd`).
- Next: above the floor (4.7 ms) the visits take 3.69 ms, lib/ntt's kernels; the rest is
  near its bounds (chirps at ~9 cycles per vector with 16 multiplies). Left: 4 huge pages
  instead of 5 (with v7's buffer, B needs blocks 0-5 only: A 4 MiB + B 3 MiB + tables 1 MiB is
  64 bytes over 8 MiB; tables into B's dead block 3, ~0.045 ms); odd lg for mid sizes.

## Sources

- Chirp z-transform: L. Bluestein, "A linear filtering approach to the computation of the
  discrete Fourier transform", 1968; L. Rabiner, R. Schafer, C. Rader, "The chirp
  z-transform algorithm", IEEE Trans. Audio Electroacoustics 17 (1969). The identity
  i j = t(i + j) - t(i) - t(j) avoids square roots of r (standard; derived here).
- Middle product: G. Hanrot, M. Quercia, P. Zimmermann, "The middle product algorithm I",
  AAECC 14 (2004) (the idea; no code read).
- Symmetric inputs halve a transform (as for the DCT): J. Makhoul, "A fast cosine transform in
  one and two dimensions", IEEE Trans. ASSP 28 (1980) (the general idea). The palindromic kernel
  from t(1 - x) = t(x) and the node relation b mod x^s - 1/z = z rev(b mod x^s - z) are derived
  here.
- Transform: our `lib/ntt` (`product.hpp`, from `problems/convolution/convolution_mod`).
