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
