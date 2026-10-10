# multivariate_convolution_cyclic

Prime p <= 10^9, K <= 18 axes, n_i >= 2, n_i | p - 1, N = prod n_i <= 2^18. Print f g mod
(x_i^n_i - 1) mod p. 10 s.
Record when opened (issue #37): 117 ms. Best judged: ours, 11 ms, clean 11:
[409664](https://judge.yosupo.jp/submission/409664) (`main.cpp` of #322). Earlier: 12 ms,
[409321](https://judge.yosupo.jp/submission/409321) (#150); 15 ms,
[409314](https://judge.yosupo.jp/submission/409314) (#146). `main.cpp` of #163: 15 ms,
[409362](https://judge.yosupo.jp/submission/409362).
I/O floor (`lib/io/notes.md`): 4.18 ms, 3.88 with fixed-width output.

Official tests: dim1 (one axis, n about 2^18, often with a large prime factor), dim2 (two axes,
e.g. 199 x 1317, 127670 x 2, 53337 x 4), max_random (7 axes of 2..10), twos (up to 18 axes of 2),
threes (2s and 3s), small, k0 (K = 0, p may be 2).

## Design

- Short axes (n <= 48): DFT of length n over F_p (root g^((p-1)/n), g a primitive root), direct
  O(n^2) on 8 lanes with Shoup products, values canonical. Axes of stride >= 8: 8 adjacent
  columns per step; a run of s columns ends with an overlapping step that stores only its new
  lanes (`vpmaskmovd`). Axes of stride < 8: rows of m words holding all of them; 8 rows at a
  time are interleaved into a buffer and transformed there.
- Long axes: the group Z_n1 x ... of the long axes as the fewest cyclic factors Z_D (CRT on the
  prime-power parts: the largest power of each prime to factor 0, the next to factor 1, ...).
  Point (i_1..) maps to y_r = sum i_j w_jr mod D_r. Then per point of the short axes' spectrum:
  Kronecker substitution with factor r padded to 2 D_r - 1, the product over Z mod three NTT
  primes below 2^28 (`../convolution_mod_1000000007/product.hpp`, `lazy::Product`: lazy
  reductions, 2^9 <= length <= 2^20, inputs < 4P raw), CRT straight to residues mod p
  (Montgomery, p odd), fold j + D_r onto j. Coefficients < 2^18 p^2 < 2^77.8, folded sums
  < 2^78.8; the primes' product is 2^83.96.
- The two factors of a product are the halves of one pair array of 2^lg words; the last prime's
  product runs in place, so the pair becomes its residues: 4 arrays of 2^lg words (pair, two
  residues, work).
- One cyclic factor (all long tests): the places are [0, D), and the CRT folds each prime's
  residues first (y_k[j] + y_k[j + D] mod P_k), then one CRT per output.
- One long axis of stride 1 (dim1, dim2_00, dim2_02): input rows go straight into pairs
  [row r of f | row r of g], each zero-padded to half the pair. The short DFTs run on the pairs
  (f and g in one pass), the products in place on each pair, and the CRT writes row r packed to
  [r n, r n + n): inside pairs < r, already consumed (n <= half a pair).
- Two long axes of coprime lengths (dim2_01): place j = r n_y + c holds the point
  ((r + a c) mod n_x, c), a = 1 / n_y mod n_x (an isomorphism Z_D -> Z_nx x Z_ny). As a matrix,
  A[r][c] = F[c][(r + a c) mod n_x] with F the rows of f along x: a transpose of rotated rows,
  done in 8 x 8 AVX2 blocks (column blocks outer; the 8 rows of F stay in L1; rows that wrap use
  masked loads and stores). The CRT writes A in place into the pair, then the scatter. Scalar for
  the last n_y mod 8 columns, and for all when x has stride > 1 or a length < 8. With no short
  axes, f, g and the result are staged in the product's own arrays (f in the pair's upper half,
  g and the result in the work array): no arrays in input order.
- Split: axes above 48 are long (the shortest dropped while the transform would exceed 2^20);
  a shorter axis joins when a cost model says so (a product 10.5 ns per transform word below
  2^18 words, 12 from there, plus 0.6 ns per word once for page faults; gather and scatter 2 ns
  per point unless direct; a short DFT 0.5 ns per element and axis length). dim2_02 (53337 x 4)
  stays direct: four products of 2^17 beat one of 2^19 plus gather and scatter.
- Pointwise product when no axis is long. The scale 1 / prod(short n) is folded into the CRT
  constants. Output `../convolution_mod/fields.hpp` (10-byte fields). `.preinit_array` start.

## Log

- 2026-10-09, claude (round 1). `lc-amd`, judge flags.
  - v1: naive per-axis DFT (vector for stride >= 8, scalar otherwise), long axes padded
    separately, 7 product buffers. 24/24, slowest 32.4 ms (twos_00: scalar columns of the
    first three axes); dim2_01 24.7.
  - v2: stride < 8 axes in an interleaved 8-row buffer, masked tail stores. twos_00 6.8 ms,
    max_random 7.7, threes 6.6; slowest dim2_01 23.6.
  - v3: 5 product buffers (the last prime transforms b in place, CRT into a). `judge.py bench`
    11 rounds, slowest 6: 23.71 -> 22.42 ms.
  - v4: long axes merged into cyclic factors (dim2_01: 199 x 1317 = Z_262083, transform 2^19
    instead of 2^20): dim2_01 22.2 -> 14.9 ms (`judge.py test`).
  - v5: gather and scatter by an odometer instead of offset tables (building the tables took
    3.1 ms for 2^18 points). Phases on dim1_01 (in-process, median of 5): parse 0.84, gather
    0.31, transforms 6.91, CRT 0.26, fold 0.15, scatter 0.25, format 0.19 ms.
    `judge.py bench` 11 rounds, slowest 10 cases: v2 23.69 -> v5 11.91 ms (ratio 0.500).
  - Checks: 24/24 official tests; `stress.py` 400 rounds against `brute.cpp` plus 40 larger
    rounds comparing `-DSHORT_LIMIT=1` (all long) and `-DSHORT_LIMIT=100000` (all short);
    ASan/UBSan: stress 60 rounds and 9 official cases, file and pipe input.
  - Submitted the merged `main.cpp` (#146): [409314](https://judge.yosupo.jp/submission/409314)
    AC 15 ms; [409315](https://judge.yosupo.jp/submission/409315) AC 21 ms (jitter; `lc-amd` 11.9).
- 2026-10-09, claude (round 2). `lc-amd`, judge flags; per-case medians of 11 interleaved runs
  (scratch script over `judge.py`'s runner), base = round 1's `main.cpp`.
  - Headroom: every dim test has n just below 2^18, so 2n - 1 needs 2^19 whatever the split
    (dim1_00 as 268 x 977: 268 x 2^11 = 2^19 too). Three primes are the minimum (n p^2 / 4 >
    2^60). Mod-p DFTs of length n need Rader or Bluestein for the factors 751..15373. So the
    9 transforms of 2^19 stay; the work was around them.
  - Direct product for one long axis of stride 1, CRT folded into f (no gather, scatter,
    fold, a/b copies): dim1_00 11.67 -> 11.04, dim2_00 11.04 -> 10.44 ms.
  - Folded CRT and no clearing for any single factor: dim2_01 11.83 -> 11.75.
  - Lost: gather by place order (strided reads of f, via an odometer over i_j = j mod n_j):
    dim2_01 11.75 -> 12.66, dim2_02 10.82 -> 11.56. Scatter fused into the CRT (point order
    gather kept): dim2_01 11.88 -> 12.24.
  - Phase costs on dim2_01 from variants that skip one step (wrong output): gather 0.32, scatter
    0.34, CRT 0.39 ms.
  - Narrowing fused into the sparse radix-4 first level (lg 19 is that path): dim1_00
    11.00 -> 10.87, dim2_01 11.70 -> 11.58.
  - Blocked place-order gather and scatter for two long axes, branch-free runs: dim2_01
    11.54 -> 11.30 (an earlier version with a wrap test per point: 11.63).
  - `judge.py bench`, 21 rounds, slowest 8 cases: 11.76 -> 11.38 ms (ratio 0.961). Slowest
    case now dim2_01 (11.3); dim1 10.9.
  - Checks: 24/24 official tests; `stress.py` 400 rounds plus 40 large (lc-intel, gcc 15.2,
    x86-64-v3); ASan/UBSan: stress 60 rounds and 9 official cases, file and pipe input.
  - Submitted the merged `main.cpp` (#150): [409317](https://judge.yosupo.jp/submission/409317)
    AC 17 ms; [409318](https://judge.yosupo.jp/submission/409318) AC 19 ms. Both above round 1's
    15 ms despite `lc-amd` 11.4 vs 11.8: judge jitter (see `tools/spikes.md`). Best stays 409314.
- 2026-10-10, claude (issue #156): the local transform moved to `lib/multimod` (`Bounded`); the
  AVX2 helpers are `multimod::add` etc., no longer in `detail`. The reduction state now stays in
  registers in the first level: transforms alone at lg 20, ratio 0.9945. `judge.py bench`,
  31 rounds, `lc-amd`: 0.9944. 24/24 official tests. Details: `lib/multimod/notes.md`.
- 2026-10-10, claude (issue #156): the local `fields.hpp` copy is gone; the solution includes
  `../convolution_mod/fields.hpp` (it was byte-identical). `main.cpp` changes in one comment line; the
  judge's command builds byte-identical executables from main's and this `main.cpp` (`lc-amd`).
- 2026-10-10, audit (claude): submissions not logged before. Who submitted them is not recorded.
  "clean" is the score without launch spikes (`tools/spikes.py`), where it differs.
  - `main.cpp` of #150, 2026-10-09 UTC:
    [409319](https://judge.yosupo.jp/submission/409319) 23:45 AC 19 ms, 16.9 MiB, clean 11
    (spikes on dim1_02 and dim2_00); [409320](https://judge.yosupo.jp/submission/409320) 23:46
    AC 15 ms, 16.6 MiB; [409321](https://judge.yosupo.jp/submission/409321) 23:46 AC 12 ms,
    16.5 MiB (linked from README.md before, not from here). With 409317-409318: 5/5.
  - Current `main.cpp` (#163), 2026-10-10 UTC:
    [409352](https://judge.yosupo.jp/submission/409352) 01:56 AC 17 ms, 17.3 MiB;
    [409362](https://judge.yosupo.jp/submission/409362) 01:58 AC 15 ms, 17.3 MiB.
  - New best judged: 12 ms (409321); was 15 (409314).
- Next: dim2_01 still pays ~0.5 ms for gather and scatter over dim1. The transforms (about
  6.5 ms) are the floor of this method; a gain there needs faster lib/ntt kernels.
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  11.36 → 11.25 ms (0.991). 24/24 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib, issue #156 round 2): the arena comes from `lib/mem/huge.hpp`
  (`mem::Arena`) instead of a local copy. Same instructions, other stack slots in `solve()` and
  `transform_short`; `judge.py bench`, `lc-bench`, 21 rounds: 11.50 -> 11.44 ms (noise). Official
  tests pass; ASan/UBSan on 6 official cases.
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).
- 2026-10-10, claude (round 3). Exploration files: `lc-opt-explore/multivariate_convolution_cyclic/r3`.
  Per-case times: static builds (judge flags) run on the host of `lc-bench` (EPYC 7B13), pinned,
  interleaved, median of 11 (`pcb.py`); phases: in-process `CLOCK_MONOTONIC` stamps, median of
  11 (`probe.py`). Host runs read about 1 ms less than `judge.py`.
  - v1: `lazy::Product` (primes below 2^28) instead of lib/multimod; direct rows padded to half
    the transform (the lazy product reads whole halves). Per case: dim1_00 10.35 -> 9.58,
    dim2_01 10.69 -> 10.11, dim2_02 10.09 -> 9.46 ms. `judge.py bench`, 21 rounds, 8 cases:
    11.16 -> 10.51 (0.936).
  - Phases of v1 (ms): dim1_00 read 0.73, products 6.33 (0.31 of it page faults of the product
    arrays: pre-faulted, 6.07), CRT 0.27, write 1.11; dim2_01 also gather 0.33, scatter 0.17,
    CRT 0.40.
  - dim2_02 direct (4 rows, products of 2^17) instead of joined (one of 2^19): in-process
    8.57 -> 7.67. Products 4 x 1.37 = 5.48 against 6.24: per word, 2^17 runs 10% faster (the
    arrays stay nearer L2, fewer page faults). The cost model now says so.
  - Skewed 8 x 8 transposes for two long axes (v2): gather of f and g 0.33 -> 0.34 cold (0.25
    warm), scatter 0.16 -> 0.13. Memory-bound, not compute: each column block writes 32 bytes
    into each of 199 rows of A, one RFO per line.
  - Pair layout, last prime in place (v3): one array of 2^lg words less. dim1_00 9.42 -> 9.32,
    dim2_01 9.90 -> 9.63 (5 rounds).
  - Folding residues per prime before one CRT (v4): CRT 0.333 -> 0.213 ms on dim1_00. CRT fused
    into the scatter: 0.462 -> 0.444 (its reads now go down columns of A).
  - Per case, v1 / v3 / v4: dim1_00 9.22 / 9.18 / 9.06, dim2_00 9.07 / 9.06 / 8.93, dim2_01
    9.72 / 9.62 / 9.55, dim2_02 9.15 / 8.29 / 8.25 ms.
  - `judge.py bench`, 21 rounds, 8 slowest cases, main against v4: `lc-bench` 11.43 -> 10.56
    (0.921); `lc-k68` 11.98 -> 11.15 (0.930).
  - Checks: 24/24 official tests (`lc-amd`); `stress.py` 400 rounds plus 40 large (cases with two
    coprime axes added); ASan/UBSan: stress 80 rounds and all 24 official cases, file and pipe;
    `-march=x86-64-v3`: stress 150 rounds.
  - CI (#322, merged): geomean 0.9295 (EPYC 7763 0.9269 and 0.9234, 9V74 0.9385).
  - Submitted the merged `main.cpp` (#322): [409664](https://judge.yosupo.jp/submission/409664)
    AC 11 ms, clean 11 (`tools/spikes.py`: spike on k0_01). Per case: dim1 8 / 8 / 9, dim2_00
    10, dim2_01 11, dim2_02 7 ms; round 2's submissions had 10-11, 9-10, 11-12, 10-11. Best
    judged 12 -> 11 ms.
  - Phases on `lc-k68` (Linux 6.8; ms): dim1_00 read 1.00, products 6.61, CRT 0.21, write 1.22;
    wall about 0.6 more (start, exit). Host runs there read 0.3-0.4 ms above `lc-bench`.
  - Lost: fused radix-16 top levels at 2^19 (forward: sparse level 1 and level 2 in one pass;
    inverse: level 2, level 1 and the scale), C++ intrinsics, results bit-equal to
    `lazy::Product` for every length 2^9..2^20. Three primes, hot: 5.98 -> 7.16 ms (forward fused
    alone 6.66, inverse alone 6.52). Each column touches 24 rows 128 KiB apart, which share L1 and
    L2 sets. The base's top passes per prime: forward level 1 0.080 ms, level 2 0.124 (a and b),
    inverse level 2 0.062, level 1 with the scale 0.098: 1.09 of 6.0 ms for three primes, near
    60-75 GB/s already (`r16/tb16.cpp`).
  - Lost: row blocks outer in the transposes (A rows written in 8 streams): gather 0.314 ->
    0.323, CRT and scatter 0.448 -> 0.468 ms.
  - v6: CRT in place into the pair, then the scatter from it, instead of fused: 0.442 -> 0.360 ms
    on dim2_01 (the fused CRT read down the columns of A).
  - Lost: f, g and the result streamed through a buffer, parsing and printing a batch of rows at
    a time (fixed-width fields end each batch with a newline; the checker compares tokens).
    dim2_01 on `lc-k68`: 4096 values per batch 10.59, 16384 10.14, 65536 9.99, all at once 9.95
    against 10.14 unstreamed: each `read_bulk` call parses its last tokens below 1024 one at a
    time. What helped was the smaller footprint, kept in v8.
  - v8: with two long axes and no short ones, f, g and the result are staged in the product's
    arrays (f in the pair's upper half, g and the result in the work array): 2 MiB less.
    Per case on `lc-k68`, v4 / v6 / v8: dim2_01 10.21 / 10.18 / 9.99, dim1_00 9.59 / 9.60 / 9.61.
    `judge.py bench`, 21 rounds, 8 cases, main (#322) against v8: `lc-k68` 11.13 -> 10.93
    (0.985); `lc-bench` 10.31 -> 10.15 (0.984). Checks as above (24/24, stress 400 + 40,
    ASan/UBSan stress 80 and 24 official cases, x86-64-v3 stress 150).
- Next: dim2_01 still pays ~0.4 ms over dim1 for the two gathers and the scatter (memory-bound:
  one RFO per line of A); dim2_00 was judged 10 against dim1's 8-9 once, while `lc-k68` puts it
  below dim1 (one sample). The products (6.0 ms hot for three primes at 2^19) are the rest:
  faster subtree kernels, or fewer passes without the set conflicts above (a buffered radix-16).

## Sources

- Chinese remainder theorem for cyclic groups (Z_ab = Z_a x Z_b for coprime a, b), as in the
  Good-Thomas prime-factor FFT; invariant-factor decomposition of finite abelian groups.
- Kronecker substitution for multivariate products.
- Our own code: `../convolution_mod_1000000007` (`product.hpp`, CRT), `../convolution_mod`
  (`fields.hpp`), `lib/io`.
