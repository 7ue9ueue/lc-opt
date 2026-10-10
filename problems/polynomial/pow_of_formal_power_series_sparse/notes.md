# pow_of_formal_power_series_sparse

f with K <= 10 nonzero terms, N <= 10^6, M <= 10^18; print the first N coefficients of f^M
mod 998244353. 10 s. Input is tiny; output up to 10 MB.

Best judged: none yet. Record when opened (issue #72): 43 ms.

## Tests

- small_dense_*: N = 10^6, i_k = 0 .. K - 1 (K = 4, 5, 7, 0, 3): recurrences of order w = 3, 4, 6,
  -, 2, all with slopes; dense output (9.9 MB). The slowest: 02 (w = 6).
- max_random_*: N = 10^6, K = 10, i_0 = 0, the others uniform (the smallest 26861 .. 85511): long
  taps only, output mostly zeros.
- low_deg_zero_*: i_0 >= 79919, M ~ 10^17: all zeros. low_deg_zero2_*: i_0 = 5 .. 96,
  M = 19 .. 87: sM = 435 .. 6624 zeros, then long taps.
- random_*: N = 0.39 .. 0.8 10^6, sparse output. min_K: K = 0, M > 0 (zeros). overflow_killer:
  sM = 2^32 and 2^64. examples: M = 0 and f = 0 (0^0 = 1).

## Design

- f = a x^s (1 + h): f^M = a^M x^(sM) (1 + h)^M; zero mod x^N once sM >= N, tested as
  M >= ceil(N / s) (sM can exceed 2^64). K = 0: f = 0, so f^0 = 1 and f^M = 0 otherwise.
- g = (1 + h)^M: (1 + h) g' = M h' g gives n g[n] = sum_d ((M + 1) d c_d - c_d n) g[n - d],
  g[0] = a^M, c_d the coefficients of h. `poly::sparse::Holonomic` (`lib/poly/holonomic.hpp`)
  with taps (d, (M + 1) d c_d, -c_d), M + 1 taken mod P (the coefficients of (1 + h)^M are
  polynomials in M). Taps with d >= N - sM are dropped.
- Output as in exp_of_formal_power_series_sparse: sM zeros as "0 " first, then g in chunks of
  25600 in a ring (or one array if a tap reaches back further), groups of 16 zeros as
  "0 0 ... 0 ", other groups as fixed-width fields (`fields.hpp`), `write(2)` per chunk, run from
  `.preinit_array`.
- Sources: the recurrence is the standard one for powers; the block solution is
  `lib/poly/holonomic.hpp`'s (derivations and sources in `lib/poly/notes.md`).

## Floor

`lc-amd`, small_dense_00, 01, 02, 04, whole process (judge's runner, 21 rounds, unchecked bench in
scratch), slowest case per round, medians: floor (read, fill 10^6 nonzero values, the same output
code) 6.63 ms; first version 9.45 (1.42); this round 8.86 (1.33). The gap is the solve: in
process, small_dense_02, 10^6 coefficients: 2.95 -> 2.35 ms.

## Log

- 2026-10-10, claude (round 1): first solution; `lib/poly/holonomic.hpp` tuned for slopes.
  - Checks: 35/35 official tests (`judge.py test`); `stress.py` 500 rounds against `brute.cpp`
    (N up to 10^6; short, long, mixed and block-edge taps; leading zeros; M in {0 .. 3}, up to
    10^18, -1 and 0 mod P, 2^29, 2^59, 2^63, sM just below and at N), and for N <= 64 `brute.cpp`
    against f^M by repeated squaring; `lib/poly/test.cpp` at -O2 (native and x86-64-v3) and
    ASan/UBSan.
  - Solve in process (`lc-amd`, 10^6 coefficients in chunks of 25600, medians of 11, ms;
    small_dense 04, 00, 01, 02 = w 2, 3, 4, 6), steps in `lib/poly/notes.md`:
    first version 2.55, 2.67, 2.75, 2.95; slopes after the reciprocal scale 2.24, 2.31, 2.35,
    2.52; odd reciprocals inside the kernel loop 2.00, 2.09, 2.19, 2.38; V before V' 1.98, 2.09,
    2.18, 2.36.
  - Probes (same runs): without the block-to-block dependency the first version took 2.23 ..
    2.60 (0.33 less), and its reciprocals 0.35; after this round the dependency costs nothing
    (the kernel is throughput-bound, about 390 vector ops per block at w = 6).
  - Not kept: windows of 32768 coefficients (no change).
- Next: the solve is ~2.3 ms of 8.9. Op counts per block at w = 6: state part 112 (V and V',
  12 columns), triangle 118 (46 `vpmuludq`; 40 in a layout of consecutive qwords, 34 at best),
  odd reciprocals ~60, scale 40, reductions 30.
