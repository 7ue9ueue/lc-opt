# log_of_formal_power_series_sparse

f with K <= 10 nonzero terms (f[0] = 1), N <= 10^6; print the first N coefficients of log(f)
mod 998244353. 10 s. Input is tiny; output up to 10 MB.

Best judged: none yet. Record when opened (issue #71): 38 ms.

## Tests

- small_dense_*: N = 10^6, i_k = k (K = 5, 6, 8, 1, 4): recurrences of order K - 1 = 4, 5, 7, 0,
  3, dense output (9.9 MB). The slowest: 00, 01, 02 (02 has w = 7).
- max_random_*: N = 10^6, K = 10, indices uniform (the smallest 36000 .. 85000). g is nonzero only
  at sums of indices; output 2 MB, mostly zeros.
- random_*: N = 0.39 .. 0.8 10^6, sparse output; small_N, min_K (K = 1, log 1 = 0): tiny.

## Design

- f g' = f' with G = n g: G[n] = n f[n] - sum_k a_k G[n - i_k] (a_0 = 1). This is 1/f's
  recurrence with r = n f[n] at the i_k: `poly::sparse::Recurrence` (`lib/poly/sparse.hpp`), the
  same kernel as inv_of_formal_power_series_sparse.
- g[n] = G[n] / n: `poly::sparse::Divider` (`lib/poly/divider.hpp`, design in
  `lib/poly/notes.md`, "Sparse"). Odd n by batch inversion (4 chains of 8 lanes), even n = 2m from
  a table of 1 / (2m) filled as it goes; 30 `vpmuludq` per 16 values.
- Output as in inv_of_formal_power_series_sparse: chunks of 25600 coefficients; G in a ring of
  history + 25600 words (divided in place once the next chunk's history is saved) when every tap
  reaches back at most a chunk, else one array and a separate chunk for g; groups of 16 zeros as
  "0 0 ... 0 ", other groups as fixed-width fields (`fields.hpp`), `write(2)` per chunk, run from
  `.preinit_array`.
- Sources: Montgomery's batch inversion (cited in `lib/poly/notes.md`); the rest derived here,
  no code read.

## Floor

`lc-amd`, small_dense_00..02, whole process, slowest of the three per round (unchecked bench in
scratch, 31 rounds, medians): floor (read, fill 10^6 nonzero values, the same output code) 6.40 ms;
the recurrence only (G printed, no division) 6.91 (1.076); first version 7.53 (1.170); this round's
version 7.31 (1.157). In process, small_dense_02 (10^6 coefficients, median of 21): recurrence
0.464 ms, division 0.447.

## Log

- 2026-10-10, claude (round 1): first solution, with `lib/poly/divider.hpp` (new);
  `sparse.hpp` and `holonomic.hpp` unchanged.
  - Checks: 24/24 official tests (`judge.py test`, `lc-amd`); `stress.py` 500 rounds against
    `brute.cpp` (1/f by its recurrence, then f' (1/f) integrated; N up to 10^6, short, long,
    mixed and block-edge indices); `lib/poly/test.cpp` (Divider against G / n for n = 1 .. 300
    and random sizes up to 10^6, in place and not, random, P - 1 and mostly zero G).
  - Division, in process (small_dense_02, ms for 10^6): first version 0.58 (GCC kept the 4 chains
    rolled, their state on the stack); unrolled 0.53; lanes as one base plus constant offsets,
    odd lanes of G and of the prefixes loaded at +1 instead of shifted, the batch inverting 2x
    where the table is written (its entries for odd m, no halving) 0.50; three passes (prefix
    products, odd reciprocals into the prefix buffer, blocks) instead of finishing blocks in the
    backward pass 0.475 (forward 0.09, backward 0.13, blocks 0.245); the 32 lane totals by a
    vector product tree, one scalar inversion at the root (was 2 x 32 dependent scalar products,
    20 us in all) 0.465; above the table's end, factors widened from memory (`vpmovzxdq`)
    instead of interleaved by `vperm2i128` 0.45; pieces cut at the table's end 0.447.
  - Not kept: 8 chains (0.483 against 0.475; the forward pass took 0.11 with 64 scalar lanes);
    streaming stores for the table, the table read from a small window, odd lanes of G by shifts:
    no change (probes), so the block pass is not memory-bound. A compile-time table of 1/n:
    GCC's constexpr operation limit (2^25) stops it after 7.5 s; the judge's flags cannot raise it.
  - Zen 3 throughput (`lc-amd`, asm loops of 12 independent ops): `vpmuludq` 2 per cycle,
    `vpsrlq` 2, `vpaddq` 4, `vpblendd` 4, `vperm2i128` 1; 6 `vpmuludq` + 6 `vpsrlq` 4.15
    cycles, 6 `vpmuludq` + 6 `vpaddq` 4.28, 4 + 4 + 4 3.0. The block pass runs at ~3 IPC with 12
    `vpmuludq` per 16 values.
- Next: the division is 30 `vpmuludq` per 16 values (prefix 6, backward 12, products 12) at
  ~65% of the multiply pipes; the recurrence is inv's kernel. Fusing the products with the
  output formatting (mul-heavy against shuffle-heavy) is untested. max_random (3.5 ms, not the
  slowest) divides all of its mostly zero G.
