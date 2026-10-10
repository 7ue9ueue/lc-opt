# log_of_formal_power_series_sparse

f with K <= 10 nonzero terms (f[0] = 1), N <= 10^6; print the first N coefficients of log(f)
mod 998244353. 10 s. Input is tiny; output up to 10 MB.

Best judged: ours, 9 ms: [409639](https://judge.yosupo.jp/submission/409639) (`main.cpp` of
#196 / #300, the same code as #182); small_N_03 took 9 ms against 1 for its peers (launch spike);
clean score 6 ms (small_dense_00 and _04 6 ms, _01 and _02 5). Earlier 14 ms (409396, spike;
clean 5). `main.cpp` of #317: 10 ms (409663, spike; clean 6). Record when opened (issue #71):
38 ms.

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
  a table of 1 / (2m) filled as it goes, kept as a ring of N / 4 words; 30 `vpmuludq` per 16
  values.
- Output as in inv_of_formal_power_series_sparse: chunks of 25600 coefficients; G in a ring of
  history + 25600 words (divided in place once the next chunk's history is saved) when every tap
  reaches back at most a chunk, else one array and a separate chunk for g; groups of 16 zeros as
  "0 0 ... 0 ", other groups as fixed-width fields (`fields.hpp`), `write(2)` per chunk, run from
  `.preinit_array`. G's ring, the divider's table (1 MB) and the text (256 KB) share one 2 MiB
  huge page: one page fault.
- Sources: Montgomery's batch inversion (cited in `lib/poly/notes.md`); the rest derived here,
  no code read.

## Floor

`lc-amd`, small_dense_00..02, whole process, slowest of the three per round (unchecked bench in
scratch, 31 rounds, medians): floor (read, fill 10^6 nonzero values, the same output code) 6.40 ms;
the recurrence only (G printed, no division) 6.91 (1.076); first version 7.53 (1.170); this round's
version 7.31 (1.157). In process, small_dense_02 (10^6 coefficients, median of 21): recurrence
0.464 ms, division 0.447.

Round 2, `lc-bench`, small_dense_02, `prun` (fork to wait, files on tmpfs, medians of 41 runs, ms):
floor 5.126 (start 0.94, setup 0.06, fill 0.19 with the page fault, format 0.66, `write` 3.10,
end 0.16); main before this round 6.101 (1.190); this round 5.970 (1.165): setup 0.14 (with the
huge page's fault, 0.075), recurrence 0.458, division 0.493, format 0.664, `write` 3.118.
Division passes in memory (10^6 values, TSC): forward 0.078, backward 0.141, blocks 0.151 below
N / 2 (they write the table) and 0.087 above.

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
  - Formatting each 64-value step right after its division (a sink called from the block pass,
    probe in scratch): next + divide + format in process 1.59 ms against 1.56 for the chunk's
    division then its format; format alone 0.62. No overlap gained; not kept.
  - On the same inputs, inv_of_formal_power_series_sparse's `main.cpp` (1/f: the same
    recurrence, no division) runs 1.055 times the floor, ours 1.140 (21 rounds).
  - Merged as #182. Submitted: [409396](https://judge.yosupo.jp/submission/409396) AC 14 ms,
    12.2 MiB. small_dense_00, 01, 04 5 ms, small_dense_02 14 ms (+9 ms: the launch spike's
    signature, `tools/spikes.md`), max_random 3, all others at most 2. `tools/spikes.py` has no
    other runs of this problem to compare with, so it reports 14; not resubmitted.
- Next: the division is 30 `vpmuludq` per 16 values (prefix 6, backward 12, products 12) at
  ~65% of the multiply pipes; the recurrence is inv's kernel. max_random (3 ms, not the slowest)
  divides all of its mostly zero G.
- 2026-10-10, claude (issue #72, pow_of_formal_power_series_sparse): the bundle changed with
  `lib/poly/holonomic.hpp` (`divider.hpp` includes it for its helpers; `Divider` itself is
  unchanged). `judge.py bench` (`lc-amd`, 21 rounds) against main: 0.9940; 24/24 official tests.
- 2026-10-10, claude (round 2). Exploration files: `lc-opt-explore/log_of_formal_power_series_sparse/round2`.
  - Resubmitted main's `main.cpp` (same code as 409396):
    [409639](https://judge.yosupo.jp/submission/409639) AC 9 ms, 12.6 MiB; small_N_03 9 ms (launch
    spike, peers 1); small_dense_00 6, _01 5, _02 5, _04 6. Clean score 6 ms. The dense cases sit
    on the 5/6 ms boundary.
  - Kept: the divider's table as a ring of N / 4 words (entry m is live from index m to 2m) in
    the caller's memory, so G's ring, the table and the text share one huge page; `HugeWords`
    mapped a second 2 MiB page whose first fault took 0.075 ms. Whole process (`prun`, 41 runs,
    small_dense_02): 6.105 -> 5.989 ms. With it, `Recurrence` broadcasts the state once per step
    (logged in #69 round 2): recurrence 0.482 -> 0.459 ms (w = 7), 0.319 -> 0.313 (w = 4).
  - `judge.py bench`, `lc-bench`, 41 rounds, against main: log 0.9816; inv 0.9944; exp 0.9972
    (its stripped executable is byte-identical, as pow's and sqrt's: noise).
  - Checks: 24/24 official tests; `lib/poly/test.cpp` at -O2 and ASan/UBSan, the table starting
    as garbage; mutants caught: ring 64 words short, no cut where writes or reads wrap, g[0] != 0.
    A ring of exactly half the table's end is safe (the block that writes m reads m - ring
    first or later), so the first try's 64 spare words were dropped.
  - Not kept:
    - The division fused into the recurrence: reciprocals for the chunk first, then the products
      and table writes for each 64-value step called from the kernel loop right after the step
      (a sink), into a separate g buffer. Recurrence + products 0.776 ms, reciprocals 0.244,
      against 0.477 + 0.492 apart: slower. Both loops issue about 3 vector ops per cycle; no
      overlap to gain.
    - Two recurrence streams per loop (the chunk's halves, each matrix column loaded once for
      both): w = 3, 4, 5, 7: 0.276, 0.327, 0.374, 0.524 ns per coefficient against 0.281, 0.322,
      0.372, 0.465. Loads are not the limit.
    - Batch inversion in 8 chains of 4 qword lanes (no odd-lane shifts or blends), odd
      reciprocals as qwords, table entries interleaved by a blend instead of `vperm2i128`:
      division 0.50-0.53 ms against 0.458 in memory.
    - Chunks of 12800 (5.970 -> 6.053 ms) or 32768 (6.016): `write` and start vary; kept 25600.
  - Zen 3 (`lc-bench`, 12 independent chains per kind): `vpmulld` 2 per cycle on the multiply
    pipes, as `vpmuludq`; shifts, `vpshufd`, `vpunpck*` and `vpmovzxdq` (memory) share two other
    pipes; `vperm2i128` 1 per cycle and blocks both groups; stores 1 per cycle; 256-bit loads 2.
    So Shoup products (2 `vpmuludq` + 2 `vpmulld` per 8 lanes) only pay with a free precomputed
    quotient, which the reciprocals lack.
  - Merged as #317; CI: log 0.9830, inv 0.9921, all 5 problems 0.9950. Submitted its
    `main.cpp`: [409663](https://judge.yosupo.jp/submission/409663) AC 10 ms, 12.3 MiB;
    small_N_04 10 ms (launch spike, peers 1); small_dense_02 6, _00, _01, _04 5. Clean 6 ms.
  - Then: the table's 16-entry groups stored as m + 0..3, 8..11, 4..7, 12..15, the order
    `vpunpck{l,h}dq` leave them in, so writing them needs no `vperm2i128`; both factors of the
    products widened from memory. Division in memory 0.457 -> 0.440 ms; `judge.py bench` (41
    rounds) against #317: 0.9897. The first try loaded the even factors after the table's
    stores: wrong where the ring is short (n around 156), caught by `lib/poly/test.cpp`, not by
    the official tests or 10^6-value runs (there the same block never reads and writes one
    slot). Mutants caught: runs not swapped, runs read at m + 0, 4, factors loaded after the
    stores.
- Next: 0.84 ms over the floor's 5.13 (recurrence 0.35 more than the floor's fill, division
  0.49 before the reordered table; both at about 3 vector ops per cycle). The division needs
  2.5 Montgomery products per value; fewer products (pairs of odd n by finite differences:
  2.25) save at most 3%.
