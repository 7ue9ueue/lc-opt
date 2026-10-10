# pow_of_formal_power_series_sparse

f with K <= 10 nonzero terms, N <= 10^6, M <= 10^18; print the first N coefficients of f^M
mod 998244353. 10 s. Input is tiny; output up to 10 MB.

Best judged: ours, 10 ms: [409705](https://judge.yosupo.jp/submission/409705) and
[409706](https://judge.yosupo.jp/submission/409706) (`main.cpp` of #344), each with one launch
spike (10 ms against 1-2 for its peers); clean score 7 ms (`tools/spikes.py`), small_dense at
6-7 ms. Record when opened (issue #72): 43 ms.

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
  polynomials in M). Taps with d >= N - sM are dropped. Every short tap has a slope; the
  kernels take them as V(n) = V0 + n V1, advanced by additions (`lib/poly/notes.md`, "Sparse"):
  the block kernel for w <= 5, the chained kernel for w = 6 .. 8.
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

Round 2, `lc-bench` (3.43 GHz), same cases, judge's runner, 31 rounds, medians of the slowest case
per round: floor 5.69 ms (each case 5.40-5.44); main 7.71 (1.36; small_dense_02 7.66); this
round 7.49 (1.32; small_dense_02 7.48). The gap is still the solve, 1.92-2.23 ms in process.

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
  - Merged as #196. CI: exp sparse 0.9713, log sparse 0.9994; all 2 0.9853. Submitted:
    [409425](https://judge.yosupo.jp/submission/409425) AC 16 ms, 13.8 MiB: small_dense_02 15
    and 04 16 ms against 6 for 00 and 01, example_00 9 ms (three launch spikes); resubmitted
    [409426](https://judge.yosupo.jp/submission/409426) AC 13 ms, 14.3 MiB: small_dense 6-7 ms,
    max_random_01 13 (a spike; its peers 3). `tools/spikes.py`: clean 7 ms for both; a clean run
    has P = 0.15 (35 cases within 9 ms of the max), so not resubmitted again.
- Next: the solve is ~2.3 ms of 8.9. Op counts per block at w = 6: state part 112 (V and V',
  12 columns), triangle 118 (46 `vpmuludq`; 40 in a layout of consecutive qwords, 34 at best),
  odd reciprocals ~60, scale 40, reductions 30.
- 2026-10-10, claude (issue #71, log_of_formal_power_series_sparse round 2): the bundle changed
  with `lib/poly/sparse.hpp` (`Recurrence` broadcasts its state once per step; unused here). The
  stripped executable is byte-identical (judge's command, `lc-amd`).
- 2026-10-10, claude (round 2): slopes without products in `lib/poly/holonomic.hpp`. Files:
  `lc-opt-explore/pow_of_formal_power_series_sparse/round2`.
  - Main re-measured (`lc-bench`, in process, 10^6 coefficients, chunks of 25600, medians of
    11 x 3-5 interleaved rounds, ms): pow w = 2, 3, 4, 6: 2.00, 2.10, 2.20, 2.38; the chained
    kernel with slopes (M' H products) 2.54 flat in w.
  - W = V(n) S with V(n) = V0 + n V1 kept in registers and advanced by 16 V1 per block (block
    kernel) or M(n) = M0 + n M1 by 8 M1 per block of 8 (chained kernel): 3 additions per vector
    of 8 instead of 4w products (block) or 16 (chained) per block. Chained with slopes 2.54 ->
    2.24, now ahead of the block kernel from w = 6 (w = 5: 2.24 both). Block kernel with slopes:
    w = 2, 3, 4, 5, 6: 2.02, 2.13, 2.23, 2.31, 2.40 -> 1.96, 2.06, 2.16, 2.24, 2.37; its
    inverter half step before the triangle (was after it with slopes): w = 2, 3, 4 -> 1.94, 2.04,
    2.16 (w = 5: 2.27). Placing the M(n) update before, after the products or at the end of the
    block: equal (2.24-2.26).
  - Result (`lc-bench`, in process): pow w = 2, 3, 4, 6: 2.00, 2.10, 2.20, 2.38 -> 1.92, 2.02,
    2.14, 2.23; sqrt w = 8: 2.51 -> 2.22; exp w = 3, 7 unchanged (1.89, 2.00).
  - Whole process (`lc-bench`, small_dense, 31 rounds): 7.71 -> 7.49 ms (paired 0.972), floor
    5.69. `judge.py bench` (21 rounds, 4 slowest cases): pow 0.9808, sqrt 0.9610, exp 0.9907,
    log 1.0079 (its stripped executable is byte-identical: noise).
  - Probe: without the batch inversion (outputs wrong) the kernels take 0.18 (w = 2) to 0.31 ms
    (chained) less. The reciprocals are 9-14% of the solve.
  - Not kept:
    - Formatting each block of 16 inside the kernel loop (a sink called by `next`, overlapping the
      text with the next blocks' chain): in process (solve and text, ms) w = 3 2.75 -> 3.03,
      w = 6 2.91 -> 3.19-3.33 (sink inlined, called mid-iteration or at its end; not inlined:
      3.26). The fused loop runs at about 2.3 ops per cycle against 2.5 and 2.75 apart.
    - G = F H of a block deferred to the next block, after its chain, from H stored and broadcast
      from memory (`vpbroadcastq`, no shuffles): chained 2.26 -> 2.41.
  - Checks: 35/35 official tests (`judge.py test`; exp 25/25, log 24/24, sqrt 45/45);
    `stress.py` 300 rounds each for pow, sqrt and exp; `lib/poly/test.cpp` at -O2 with default
    kernels, `-DHOLONOMIC_CHAINED=0` and `1`, and ASan/UBSan; 8 mutations fail it (list in
    `lib/poly/notes.md`).
  - Merged as #344. CI: pow 0.9639 (EPYC 9V45 0.9686, 7763 0.9738, Xeon 6973P 0.9494), sqrt
    0.9551, exp 1.0001, log 1.0067 (identical executable); all 4 0.9812. Submitted:
    [409705](https://judge.yosupo.jp/submission/409705) AC 10 ms, 14.1 MiB: low_deg_zero_01
    10 (spike; peers 2), small_dense_04, 02 7 ms, 01, 00 6; resubmitted
    [409706](https://judge.yosupo.jp/submission/409706) AC 10 ms: random_03 10 (spike; peers 1),
    small_dense_04, 02, 00 7, 01 6. `tools/spikes.py`: clean 7 ms for both (P(clean run) = 0.15);
    not resubmitted again. The judge rounds the 0.1-0.2 ms gained per case away.
- Next: the solve is 1.9-2.2 ms of 7.5 on `lc-bench` (floor 5.4-5.7). The chained kernel runs
  at about 123 cycles per 16 coefficients against a chain of about 72 and about 80 cycles of
  vector ops; the batch inversion takes 0.18-0.31 ms. A 6 ms judged score needs about 0.3-0.5 ms
  more (a guess from the 6-7 ms spread of small_dense).
