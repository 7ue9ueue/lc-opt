# lcm_convolution

N <= 10^6, a_i, b_i < 998244353; print c_k = sum over lcm(i, j) = k of a_i b_j for k = 1..N. 5 s.
Large tests: N = 10^6 (max_random), near primes and near prime squares; ~20 MB input, 10 MB output.

Record when opened: 37 ms. Best judged: none yet.

## Design

- c = Moebius(zeta(a) zeta(b)), zeta = sums over divisors. Per prime p: zeta x_ip += x_i (i
  ascending), Moebius x_ip -= x_i (i descending). The passes commute.
- Same skeleton as `../gcd_convolution` (read for ideas, code written here): pairs [a | b R] so one
  load fetches both, b pre-scaled by R = 2^32 for one Montgomery reduction per product; one
  mapping in 2 MiB pages (pairs, then b, later c, then the output text).
- a is parsed into the second half of the pair array, so the interleave runs upwards (pair k
  overwrites only a values below k). Zeta 3 is fused with it, target-contiguous: 24 targets per
  step take 8 sources k / 3 from below (permute + blend).
- Zeta 5..13: scalar passes. Zeta 2 + product + Moebius 2 in one ascending sweep:
  A_k += A_k/2, c_k = A_k B_k - A_k/2 B_k/2 (the second product recomputed), 8 targets per step.
- Moebius 3..13: scalar passes on c.
- Primes >= 17 together: x_im += x_i over m coprime to 30030 (wheel, 5760 spokes). Zeta: target
  segments of 2^15 pairs descend, sources (below the segment) are still old; segment 0 by source,
  descending. Moebius: x_im -= final x_i, segment 0 by source ascending, then segments of 2^16
  dwords ascending. In a segment, m <= 2048 go by m over a run of sources; larger m by source
  i <= n / 2049 over a run of m.
- Runs from `.preinit_array`, ends with `_exit`. Output: `../convolution_mod/fields.hpp`.

## Log

- 2026-10-09, claude, round 1. `lc-amd` (EPYC 7B13), judge image and flags.
  - v1 (this `main.cpp`): 29/29 official tests, slowest 16.4 ms (`judge.py test`). Stress
    300 rounds (`stress.py`: brute N <= 3000, per-prime reference up to N = 300000). ASan/UBSan
    (GCC 15.2, -O1): 29/29 tests plus pipe input.
  - Phases, max_random_00, medians of 21 (ms): parse 3.53, interleave + zeta 3 0.48, zeta
    5..13 0.63, rough zeta 1.99, zeta 2 + product + Moebius 2 0.60, Moebius 3..13 0.56, rough
    Moebius 1.96, output 3.75. Sum 13.5; compute 6.2.
  - The rough sweeps: by m 0.95 ms (~0.92 N contributions), by source 1.15 ms (~1.1 N), so
    ~3.5 cycles per contribution. `perf` on `lc-intel`: the samples sit on the target loads.
  - No gain (all probe medians, same run as a v1 control):
    - L1 sub-blocks (16 KiB) for dense generators (m <= 64..256, sources <= 16..64) inside each
      L2 segment, stateless bounds: rough zeta 2.09-2.36 vs 1.99; rough Moebius 1.82-1.95 vs 1.96.
    - Segments of 2^13, 2^14, 2^16 pairs (zeta) / 2^14, 2^15, 2^17 dwords: 2^15 / 2^16 best.
    - `prefetchw` 4..32 spokes ahead in the by-source loops: +0.25 ms each sweep.
    - Keeping P in a register (GCC reloaded the constant per contribution): rough zeta 1.98 vs
      1.99. A `cmov` modular subtract for c: rough Moebius 2.29 vs 1.95.
    - Zeta 5..13 fused into the interleave by inclusion-exclusion (F_k = x_k + sum over
      squarefree d of -mu(d) F_k/d; d = 3 by vectors, the rest by source per 1536-pair block), and
      Moebius 3..13 likewise fused into the product sweep (products written as dwords over the
      dead pairs, c into its own array). Small-prime total 2.23-2.27 ms vs 2.26 for any number of
      fused primes: the extra terms (Σ 1/d = 1.15 N for all five primes vs 0.84 N) cost what the
      saved sweeps did.
    - Not storing the pairs above n / 2 in the product sweep: 0.596 vs 0.603 ms.

## Next

- The rough sweeps (~4 ms) are the largest compute. Each contribution is a read-modify-write of
  a separate line in L2; neither L1 blocking nor prefetching helped. Untested: fewer
  contributions by staging (primes 17..1000, then > 1000; gcd notes count 1.93 N vs 2.14 N).
- I/O (parse 3.5 ms, output 3.75 ms) belongs to `lib/io` and the formatter.

## Sources

- Design from `../gcd_convolution/notes.md` (zeta/Moebius by per-prime passes, wheel sweep,
  Montgomery pairs); dual direction worked out here.
- `.preinit_array` start: `../convolution_mod/solution.cpp`.
