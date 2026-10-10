# multipoint_evaluation_on_geometric_sequence

f of N coefficients, M, a, r mod 998244353, N, M <= 2^19; print f(a r^i) for i < M. 10 s.
Slowest tests: max_random_00..02 (N = M = 2^19, random a, r). The a0_r01 tests have a = 0 or
r in {0, 1}: no product (2.5-4.7 ms). near_pow_of_2 has N, M near 2^18; nm_1 has N or M = 1.

Best judged: ours, 10 ms: [409475](https://judge.yosupo.jp/submission/409475) (`main.cpp` of #227),
clean (`tools/spikes.py`).
Record when opened (issue #76): 34 ms.

## Design

- Chirp z-transform (Bluestein): with t(k) = k (k - 1) / 2, i j = t(i + j) - t(i) - t(j), so
  f(a r^i) = r^-t(i) sum_j A_j r^t(i + j) with A_j = c_j a^j r^-t(j). A middle product: one
  cyclic convolution of length L >= N + M - 1 of A (at [0, N)) and the reversed chirp
  B_u = r^t(L - 1 - u) (all of [0, L)) gives f(a r^i) = r^-t(i) (A B)[L - 1 - i], so the
  outputs sit at the top end in reverse order, aligned to vectors for any M.
- Chirps: `lib/poly/chirp.hpp` (new): x_k = c s^k q^t(k), 32 per step, x_(k+32) = x_k g_k with
  g_(k+32) = g_k q^1024, one Montgomery and one Shoup product per vector.
  B_u = r^t(L-1) (r^-(L-2))^u r^t(u).
- The convolution is convolution_mod's (`ntt::detail::Subtrees` and `forward_radix8` from
  `lib/ntt/product.hpp`, moved there from convolution_mod by issue #156) with two
  changes: B fills the whole length, so its first pass is a full radix-8 one (u = f_lo + f_hi,
  v = f_lo - f_hi, then the two radix-4 groups); and the last radix-2 level computes only the
  upper half, (u - w) [L/2 - 1 - i], times r^-t(i) and the transform's scale, reversed.
- L = 2^lg >= 2 max(N, M) with lg even (convolution_mod's 2 * 4^j vector layout), so A and the
  outputs each lie in one half. Odd lg rounds up: near_pow_of_2 (N, M ~ 2^18) uses 2^20, not
  2^19 (8.5 ms against 10 for max_random; the score is unaffected).
- a = 0 or N = 1: c_0 everywhere (only c_0 is read). r = 0, r = 1 or M = 1: f(a) by Horner's
  rule in x^32 over 4 vectors of lanes, then c_0 or f(a).
- `lib/io` input (`io::read_bulk`) into the transform buffer; output in 10-byte fixed-width
  fields (`problems/convolution/convolution_mod/fields.hpp`, judge-specific). One mapping in
  huge pages for A, B, the tables and the text. The program runs from `.preinit_array` and
  ends with `_exit`.

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

## Sources

- Chirp z-transform: L. Bluestein, "A linear filtering approach to the computation of the
  discrete Fourier transform", 1968; L. Rabiner, R. Schafer, C. Rader, "The chirp
  z-transform algorithm", IEEE Trans. Audio Electroacoustics 17 (1969). The identity
  i j = t(i + j) - t(i) - t(j) avoids square roots of r (standard; derived here).
- Middle product: G. Hanrot, M. Quercia, P. Zimmermann, "The middle product algorithm I",
  AAECC 14 (2004) (the idea; no code read).
- Transform: our `lib/ntt` (`product.hpp`, from `problems/convolution/convolution_mod`).
