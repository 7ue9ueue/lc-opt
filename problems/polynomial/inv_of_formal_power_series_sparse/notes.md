# inv_of_formal_power_series_sparse

f with K <= 10 nonzero terms (i_0 = 0, a_0 != 0), N <= 10^6; print the first N coefficients of
1/f mod 998244353. 10 s. Input is tiny; output up to 10 MB.

Best judged: ours, 10 ms: [409344](https://judge.yosupo.jp/submission/409344) and
[409351](https://judge.yosupo.jp/submission/409351) (`main.cpp` of #159); 409351 has one launch
spike, its clean score is 5 ms. Earlier: 13 ms, [409329](https://judge.yosupo.jp/submission/409329)
(same version, clean 5 ms). Record when opened (issue #69): 24 ms.

## Tests

- small_dense_*: N = 10^6, i_k = k (K = 5, 6, 8, 1, 4): order-7 recurrences at most, dense
  output (9.9 MB). The slowest (small_dense_02, w = 7).
- max_random_*: N = 10^6, K = 10, indices uniform (the smallest near 70000). g is nonzero only
  at sums of indices: a few hundred values, output 2 MB of zeros.
- min_K, random, small_N: smaller.

## Design

- g[n] = [n = 0] / a_0 + sum_k (-a_k / a_0) g[n - i_k]: `poly::sparse::Recurrence`
  (`lib/poly/sparse.hpp`, design in `lib/poly/notes.md`). Taps below 16 go through a 64-row jump
  matrix from the last w values (w products per coefficient, no dependency within 64), longer
  taps through 16-wide vector loads of earlier values.
- Chunks of 25600 coefficients: solve, format, `write(2)` (256 KB blocks skip the Writer's
  buffer). If every tap reaches back at most a chunk, the coefficients live in a ring of
  history + 25600 words; else in one array. Ring and text share one huge page.
- Output (judge-specific: the checker compares tokens): groups of 16 zeros as "0 0 ... 0 "
  (32 bytes), other groups as fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  its pipelined loop on each run of nonzero groups).
- Runs from `.preinit_array`, ends with `_exit` (as convolution_mod).

## Floor

`lc-amd`, small_dense_02, in-process phases (ms, median of 21 runs): solve 0.53, format 0.66,
`write()` of 9.9 MB 2.93; total 4.12. Whole process (`tools/judge.py test`) 5.8 ms.
Output-only floor (read, fill 10^6 values, fixed-width fields in chunks): same run, 21 rounds,
slowest of small_dense_00..02: floor 6.45, ours 6.83 (ratio 1.066; lc-amd ran ~15% slow then).
The gap is the solve.

## Log

- 2026-10-09, claude (round 1): first solution, with `lib/poly/sparse.hpp` (new).
  - Checks: 24/24 official tests; `stress.py` 500 rounds against `brute.cpp` (N up to 10^6;
    short, long, mixed and block-edge taps; ring and array modes); ASan/UBSan on all official
    cases, file and pipe input; `lib/poly/test.cpp` (recurrence against its definition, the
    reduction at its bounds; a mutation of the unfolded-sum bound fails it).
  - Steps, small_dense_02 solve phase in process (ms): sums in a struct passed by reference,
    one call per block of 16 (GCC kept the sums on the stack) 2.16; sums in registers, 32 per
    step 0.85; chunked solve and output with a ring 0.80 (and max_random 4.8 -> 1.8 ms total
    from zero groups); 64 per step, kernels unrolled per width with an empty asm after each
    column (GCC formed all products first and spilled 14 of them per block) 0.68; the block
    holding the next state first 0.53.
  - Kernel in memory (ns per coefficient, ring of 25600, `lc-amd`): w = 1: 0.175, 3: 0.269,
    7: 0.463, 12: 0.724, 15 (folded sums): 0.938. Per column 2.8 cycles per 16 values, the
    measured cost of 4 `vpmuludq` + 4 `vpaddq` (they share pipes; lib/poly/notes.md).
  - Not kept: state in registers broadcast by `vpermd` (w = 7: 0.711 at 32 per step, 0.766 at
    64; GCC spilled the state anyway); 4 KiB pages for the ring and text (whole process 1.019);
    chunks of 12800 or 6400 with direct `write(2)` (within noise, ±2%; 6400 through the Writer
    +5%, it copies blocks below 64 KiB).
  - Each column of the kernel as asm in the order M M A A (lib/poly's leaf-product finding):
    w = 7 0.480 vs 0.466 ns per coefficient; not kept.
  - Submitted the merged `main.cpp` (#159): [409329](https://judge.yosupo.jp/submission/409329)
    AC 13 ms, 10.3 MiB. small_dense_00 13 ms (launch spike), small_dense_01 4, _02 5, _04 5;
    all other cases at most 3. Clean score 5 ms.
- 2026-10-10, audit (claude): submissions of `main.cpp` (#159) not logged before; who submitted
  them is not recorded. 2026-10-10 UTC, 10.5-10.6 MiB:
  [409340](https://judge.yosupo.jp/submission/409340) 01:47 AC 14 ms, clean 10 (spike on
  small_dense_04); [409344](https://judge.yosupo.jp/submission/409344) 01:48 AC 10 ms (example_00
  10 ms, the rest at most 5; `tools/spikes.py` flags nothing);
  [409351](https://judge.yosupo.jp/submission/409351) 01:54 AC 10 ms, clean 5 (spike on
  small_N_02). With 409329: 4/5. New best judged: 10 ms (was 13, 409329).
- Next: the solve is at the products' pipe bound; what is left is format (0.66 ms, shared
  `fields.hpp`) and `write()` (2.9 ms, kernel). max_random's long-tap path (0.9 ms) could skip
  zero sources; it is not the slowest case.
