# exp_of_formal_power_series_sparse

f with K <= 10 nonzero terms (f[0] = 0), N <= 10^6; print the first N coefficients of exp(f)
mod 998244353. 10 s. Input is tiny; output up to 10 MB.

Best judged: ours, 7 ms: [409367](https://judge.yosupo.jp/submission/409367) (`main.cpp` of
#169); no launch spike, clean score 7 ms. Record when opened (issue #70): 39 ms.

## Tests

- small_dense_*: N = 10^6, i_k = 1 .. K with K = 4, 5, 7, 0, 3: recurrences of order K, dense
  output (9.9 MB). The slowest: 00, 01, 02.
- max_random_*: N = 10^6, K = 10, indices uniform (the smallest 36000 .. 85000). g is nonzero only
  at sums of indices: 100 .. 850 values, output 2 MB of zeros.
- random_*: N = 0.39 .. 0.8 10^6, K = 10, 10, 10, 1, 2, sparse output; small_N, min_K (K = 0):
  tiny.

## Design

- g' = f' g: n g[n] = sum_k i_k a_k g[n - i_k], g[0] = 1. `poly::sparse::Holonomic`
  (`lib/poly/holonomic.hpp`, design in `lib/poly/notes.md`, "Sparse"): blocks of 16 coefficients
  G = F (y ⊙ (V S)) mod x^16 from the last w = max i_k values S, with F = exp(f) mod x^16 and
  y[t] = 1 / (n + t). The constant-coefficient jump of inv_of_formal_power_series_sparse does not
  apply: the coefficients depend on n.
- Output as in inv_of_formal_power_series_sparse: chunks of 25600 coefficients in a ring, groups
  of 16 zeros as "0 0 ... 0 ", other groups as fixed-width fields (`fields.hpp`), `write(2)` per
  chunk, run from `.preinit_array`.

## Floor

`lc-amd`, small_dense_00..02, whole process (judge's runner, 21 rounds, unchecked bench in
scratch): floor (read, fill 10^6 nonzero values, the same output code) median 6.48 ms (min 5.33),
ours 8.69 (min 7.42): 1.36 times the floor. In process, small_dense_02 (11 runs): setup 0.13,
`next` 2.36, format 0.65, `write` 3.4; total 6.6 ms. The gap is the solve.

Round 2, `tools/speed.py bench` (`lc-bench`, 11 rounds, medians, ms; its floor writes the output
bytes without formatting them): small_dense_00, 01, 02, 04: 7.19, 7.25, 7.18, 7.05 against
floor 4.63, 4.42, 4.42, 4.40 (1.55-1.64); max_random 3.50-3.71 against 1.41-1.45 (2.5-2.6, not
the slowest: every block still takes reciprocals and the long-tap path).

## Log

- 2026-10-09, claude (round 1): first solution, with `lib/poly/holonomic.hpp` (new).
  - Checks: 25/25 official tests; `stress.py` 500 + 300 rounds against `brute.cpp` (N up to 10^6;
    short, long, mixed and block-edge taps); ASan/UBSan on all official cases, file and pipe
    input (50 outputs accepted); `lib/poly/test.cpp` sparse tests at -O2 and ASan/UBSan.
  - Steps (solve of 10^6 coefficients in chunks, w = 7, in a warm loop; kernel details in
    `lib/poly/notes.md`): 3.19 ms (state and H through memory, latency-bound) -> 2.93 (registers)
    -> 2.74 (`vperm2i128` + `vpshufd` broadcasts) -> 2.56 (upper half first) -> 2.24 (even
    reciprocals from a half-size table). In a fresh process the table's page faults cost 0.9 ms
    until it moved to huge pages.
  - `judge.py test`, small_dense: 8.4 ms (first version) -> 7.3 .. 7.7 ms.
  - Not kept: the triangle's two multiplies of each column before its two adds (order M M A A,
    an empty asm between): w = 7 2.245 against 2.213 ms, w = 3 2.081 against 2.089.
  - Merged as #169. Submitted: [409367](https://judge.yosupo.jp/submission/409367) AC 7 ms,
    14.3 MiB. small_dense_00, 01, 02, 04 7 ms; max_random 3; all others at most 2. No spike
    (`tools/spikes.py`: clean 7 ms). inv_of_formal_power_series_sparse's small_dense: 4-5 ms.
- Next: the solve (2.36 ms in process) is ~260 vector ops per block of 16, throughput-bound
  (1.975 ms without the dependency between blocks, 2.21 with it). Ideas: the triangle in a
  consecutive-qword layout (40 instead of 46 `vpmuludq`), multiplies and adds in the order
  M M A A (asm), the odd reciprocals (~0.25 ms) interleaved with the kernel. max_random (3.5 ms,
  not the slowest) computes reciprocals for all blocks though g is mostly zero.
- 2026-10-10, claude (issue #72, pow_of_formal_power_series_sparse): `lib/poly/holonomic.hpp`
  changed there; for exp the odd reciprocals now run inside the kernel loop, a half step of the
  next window's batch inversion per block (design in `lib/poly/notes.md`). In process (`lc-amd`,
  10^6 coefficients, w = 7 and w = 3): 2.28 -> 2.10 ms, 2.15 -> 1.90. `judge.py bench`
  (`lc-amd`, 21 rounds) against main: 0.9627. 25/25 official tests; `stress.py` 200 rounds.
- 2026-10-10, claude (round 2): a second kernel in `lib/poly/holonomic.hpp`, chained blocks of 8
  (design, probes and the attempts that lost in `lib/poly/notes.md`, "Sparse" and the log entry
  for #70 round 2). Files: `lc-opt-explore/exp_of_formal_power_series_sparse`.
  - Main re-measured first (`lc-bench`, in process, 10^6 coefficients, ms): w = 3, 4, 5, 7:
    1.95, 2.00, 2.05, 2.14. The block kernel is throughput-bound (the same time without the
    dependency from block to block) and runs at about 1 `vpmuludq` per cycle; 6 fewer products
    per block (consecutive-qword triangle) changed nothing.
  - Chained kernel: the next block's W = M H from this block's H (M = V T, 8 x 8), so the chain
    from block to block skips the triangle; broadcasts within 128-bit halves. 2.01 ms for every
    w <= 8, latency-bound (1.77 without the dependency). Used from w = 5; below, the block kernel
    with the inverter's half step before the triangle: 1.91 (w = 3), 1.98 (w = 4).
  - Slowest case in process: 2.14 -> 2.01 ms (w = 7). `judge.py bench` (`lc-bench`, 21 rounds,
    small_dense_00, 01, 02, 04): 7.45 -> 7.36 ms (0.9845).
  - Checks: 25/25 official tests (and log 24/24, pow 35/35, sqrt 45/45: their bundles changed);
    `stress.py` 300 rounds (exp, sqrt, pow); `lib/poly/test.cpp` at -O2 with the default choice,
    `-DHOLONOMIC_CHAINED=0` and `1`, ASan/UBSan (default and `1`); new tests at the chained
    kernel's widest (8 taps at P - 1, with and without slopes); mutations of the chained kernel
    fail them.
