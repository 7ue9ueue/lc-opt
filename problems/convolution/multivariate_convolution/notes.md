# multivariate_convolution

K <= 18 variables, n_l >= 2, N = prod n_l <= 2^18; print f g mod (x_1^n_1, ..., x_K^n_K) mod
998244353. 10 s.

Record when opened: 117 ms (issue #36). Best judged: ours, 14 ms:
[409322](https://judge.yosupo.jp/submission/409322) (round 2 `main.cpp`).

Official tests (shapes): twos (18 and 17 variables of size 2), threes (14 and 13 variables of
sizes 2 and 3), max_random (7 variables of 2..10, N = 151200 and 181440), dim1 (N ~ 2^18), dim2,
small, k0.

## Design

`solution.cpp` picks one of three methods: ranked if it fits, else split or graded by a cost model.

- Graded (any shape; used on dim1, dim2). chi(i) = sum_{j<k} floor(i / P_j),
  P_j = n_1 ... n_j. A carry out of digit j adds 1 to chi(i + j) - chi(i) - chi(j); the linear
  product has no carry out of the top digit, so a term has 0..k-1 carries. With grades mod m >= k,
  m | P - 1 (1, 2, 4, 7, 8, 14, 16, 17, 28, 32), t -> w^s splits the graded product into m plain
  cyclic convolutions of length 2^ceil(log2(2N - 1)) (`ntt::Convolution`, reused m times):
  c_i = (1/m) sum_s w^(-s chi_i) [(f w^(s chi)) * (g w^(s chi))]_i. Weights come from a table by
  `vpermd` on chi mod m (u8 per position). m = k is optimal for one grading mod m: k - 1 carry
  positions can carry in any subset, so m < k would let a nonempty subset sum to 0 (Davenport
  constant of Z_m is m).
- Ranked (every n_l <= 3 and at least three n_l = 2; used on twos and threes). Each variable is
  evaluated at n_l points: {0, 1} or {0, 1, -1}. A rank t^|d| tracks total degree, products
  truncated at t^R, R = sum (n_l - 1). Evaluation reduces x^d, d >= n_l, to lower powers, so a
  wrapped term sits at a rank above its position's degree and is never read. This is the ranked
  zeta transform of subset convolution, extended to size 3.
  - Digit order: three size-2 variables first: the 8 lanes of a vector, evaluated in registers
    (shuffle, blend, add). Then the block: the next variables up to Nb positions such that the
    R + 1 ranks of f and g fit 320 KiB. Ranks are stored position-major ([u][r]), so a transform
    along a block variable is one pass over blocks of R + 1 vectors.
  - Top variables (above the block): for each top point t, the block is evaluated at t straight
    from the rows of f and g it depends on (3^T2 7^T3 row additions in all, signed, per top rank),
    multiplied, interpolated, and added into the rows of c that depend on t. Nothing larger than
    a block is ever materialized per rank.
  - Pointwise: per vector, only ranks r with nz(p) <= r <= min(R, 2 cap(p)) are formed (cap: sum of
    n - 1 over nonzero coordinates, bounds the ranks present; nz: number of nonzero coordinates,
    bounds the ranks a position reads). Products accumulate in 64-bit lanes (`vpmuludq`), folded
    once after 16 terms, then one Montgomery reduction (2^-32 undone at the end).
  - Size 3 interpolation returns twice the coefficients (2 a0 = 2 F0, 2 a1 = F1 - F2,
    2 a2 = F1 + F2 - 2 F0); for top variables the 2 F0 is folded into f's evaluation as a
    factor 2 per zero coordinate, so every top row operation is a plain signed add.
  - Block transforms (round 2): two variables per step in registers (forward always, inverse only
    for 2 x 2), each group only on its rank range: forward, the ranks that can be nonzero
    ([digit sum above the step, + n - 1 of the step and of nonzero point coordinates below,
    + top cap + 3]); inverse, the ranks read later ([digit sum below, + n - 1 from the step on,
    + planes + 2]). Ranges per step and vector in u8 tables.
  - Pointwise (round 2): four ranks at a time per parity (even lanes, then odd), so each f rank is
    loaded once per four products and the 4 sums stay in registers; g is zero-padded by 3.
  - Spread and gather (round 2): lane v of rank r is top plane r - sum(u) - popcount(v): three
    `vpblendd` per rank instead of masked OR chains through memory. Lane evaluation by shifts
    (`vpsllq`, `vpslldq`, `vperm2i128`) instead of shuffle + blend.
- Split (round 2; any shape; used on max_random, small). Outer variables O (Q = prod n_o
  positions) by schoolbook over their digits, the inner ones by the graded method with
  m = grade_modulus(|inner|) grades: for each grade, 2Q forward transforms of length
  2^lg >= 2 N/Q - 1, the products h_e = sum over a + b = e (digitwise, no carry) of f_a g_b,
  Q inverse transforms. Fewer inner variables give fewer grades (k' <= 4: m = 4).
  - Transform (`transform.hpp`): lib/ntt's tables and radix-4 kernels down to nodes of 8 vectors,
    then per node three levels across vectors, an 8 x 8 transpose and three levels with per-lane
    twiddles (tables rearranged per node), down to single points. The top level is fused with the
    weighted load and the weighted, accumulated store. All 3Q arrays run depth first together, so
    the pointwise step happens per node of 8 vectors while the node is in L1.
  - Pointwise: the lowest outer digit is the largest outer size <= 5 (the block); per point and
    row pair, the block's truncated product runs in registers (even lanes, then odd lanes).
    Rows are processed in descending order, so h_e can overwrite f_e.
  - Layout: f, g and c are permuted to [outer][inner] rows once (runs of consecutive positions).
  - Planner: all multisets of outer sizes with Q <= 4096 and lg <= 15; cost = m (3 Q 2^lg lg
    0.107 + pairs 2^lg c_block + N), c_block = 0.094 ns (block 3-5), 0.13 (2), 0.19 (1); graded:
    m 2^lg lg 0.25. Constants fitted on lc-amd (log below).
- Memory: blocks of 256 KiB or more are 2 MiB aligned with `MADV_HUGEPAGE`; smaller ones take
  small pages (round 1 rounded every block to 2 MiB pages, faulting in a whole huge page for each).
- Output: `../fixed_width.hpp` (judge-specific padding). `.preinit_array` start and `_exit`.
- `-DFORCE_GRADED` forces the graded method, `-DFORCE_SPLIT` the cheapest split;
  `-DBLOCK_BYTES=1` makes every non-lane variable a top one. `stress.py` runs all three and the
  default build against `brute.cpp`.

Sources: the ranked transform is the subset-convolution technique of Björklund, Husfeldt, Kaski,
Koivisto, "Fourier meets Möbius: fast subset convolution", STOC 2007 (recalled). The chi grading
for truncated multivariate products is a known technique (recalled from competitive-programming
folklore; no code read). Montgomery reduction with `vpmuludq`, `vpermd` lookups: standard.
The split method (schoolbook outer digits around graded inner transforms) and `transform.hpp`
(lib/ntt's kernels, then a transpose and per-lane twiddles for the last three levels) are ours;
no sources read.

## Log

- 2026-10-09, claude (round 1): first solution, graded + ranked.
  - Checks: 17/17 official tests (`tools/judge.py test`, lc-intel and lc-amd); `stress.py` 400
    rounds (default, forced graded, tiny block); ASan/UBSan (`-O1`) on 9 official cases, default
    and `-DFORCE_GRADED` (covers m = 14 and 28 lookups).
  - lc-amd, judge flags, `judge.py test`: max_random_01 19.8 ms, max_random_00 19.3, twos_00 17.8,
    threes_00 15.7, threes_01 12.3, dim2 9.0, twos_01 9.0, dim1 7.3. `judge.py bench`, 21 rounds,
    slowest 5 cases: median 20.32 ms (min 19.96).
  - Phases on lc-amd (ms, in-process clock, 5 runs, scratch build):
    - twos_00 (ranked, Nb = 2^11, 7 top variables): read 0.9, evaluate top 1.2, lanes and rank
      spread 2.9, block transforms forward 3.1, pointwise 3.6, block inverse 1.5, gather 1.2,
      interpolate top 0.5, write 0.4.
    - max_random_00 (graded, k = 7, m = 7, length 2^19): read 0.6, compute 16.8, write 0.25. The 7
      convolutions are ~2.2 ms each; weights and accumulation ~1.4 ms.
  - Ranked, steps that paid (twos_00 compute, ms): planes [t][top rank][Nb] for the whole
    input, then per block -> evaluating each top point directly from rows of f: evaluate + interpolate
    6.3 -> 1.7. Rank-major blocks ([r][u], 8 KiB stride between ranks of one vector) ->
    position-major ([u][r]): spread 6.1 -> 2.9, pointwise 4.5 -> 3.7. Total compute 26 -> 15.
  - Graded: running weight products (two Montgomery products per position and step, four
    arrays) -> `vpermd` table lookup on chi mod m: 17.2 -> 16.8 ms.
  - Merged in #142. Submitted: [409313](https://judge.yosupo.jp/submission/409313), AC, 20 ms,
    21.3 MiB (record 117 ms).

- 2026-10-10, claude (round 2): split method, ranked block transforms, pointwise, spread.
  - Checks: 17/17 official tests (`judge.py test`, lc-amd); `stress.py` 600 rounds, four builds
    (lc-intel); ASan/UBSan (`-O1`) on all official cases, four builds.
  - lc-amd, judge flags, `judge.py test`: twos_00 13.4 ms (17.8), threes_00 12.7 (15.7),
    max_random_01 11.3 (19.8), threes_01 10.3, max_random_00 10.2 (19.3), dim2 9.1, dim1 7.4,
    twos_01 6.9. `judge.py bench`, 21 rounds, slowest 8 cases: round 1 20.11 ms (min 19.76), round 2
    13.84 (min 13.44), ratio 0.688.
  - Split on max_random (lc-amd, whole program, median of 7; outer sizes, Q, lg, grades):
    - max_random_00 (6 6 2 10 7 3 10): {2,3,7} 42, 2^13, 4: 10.40 ms; {2,3,10}: 14.36;
      {2,3,6} (2^14): 16.04; {3,6,6}: 16.11; {10} (2^15, 7 grades): 16.60; {2,3,6,6}: 18.20;
      {2,3,6,7}: 19.20. Graded (round 1): 19.3.
    - max_random_01 (9 4 7 6 5 3 8): {3,5,6} 90, 2^12, 4: 11.41; {3,4,8}: 12.18; {3,4,5} (2^13):
      13.79; {5,6} (7 grades): 20.82.
  - Split, steps (max_random_01 forced split, ms): transforms one by one with a per-point
    pointwise over all pieces 14.3; depth first all together 14.7 (no gain by itself); pointwise
    with the block in registers 17.4 -> `#pragma GCC unroll` (GCC -O2 left the 5 x 5 loops rolled,
    sums in memory) 13.5; transform bottoms unrolled 12.8; allocation fix 11.45; one arena 11.1.
  - Allocation: perf showed 12-15% kernel time (2 MiB page zeroing: every block, even 3 KiB, got
    its own huge page). Threshold for huge pages, lc-amd median: always (aligned) twos_00 17.04,
    threes_00 14.91; >= 256 KiB 16.84, 14.68; >= 2 MiB 18.55, 16.98. Kept 256 KiB.
  - Ranked phases on twos_00 (ms, before -> after): forward 2.5 (with pruning alone; 3.1 before)
    -> 1.4 (two variables per step); inverse 1.6 -> 0.9; pointwise 3.66 -> 3.05; spread 3.0 ->
    1.84; gather 1.14 -> 0.69. threes_00: fusing 3 x 3 pairs in the inverse 1.10 -> 1.62 (wider
    union ranges), so the inverse fuses only 2 x 2; forward fuses all (1.43 -> 1.12).
  - No gain: ranked pointwise with four ranks and both parities at once (3.66 -> 4.0: 8 sums plus a
    rotated g window spill, the sums go through memory); visiting vectors sorted by (cap, nz) for
    predictable loop bounds (3.15 -> 3.15).
  - Merged in #153. Submitted: [409322](https://judge.yosupo.jp/submission/409322), AC, 14 ms,
    14.8 MiB; [409323](https://judge.yosupo.jp/submission/409323), AC, 22 ms (same source; a judge
    launch spike, see `tools/spikes.py`).

## Next

- Ranked pointwise is the largest phase (twos_00 3.05 of ~11 ms compute): a third is the per-rank
  reduction; lanes widen each vector's cap by 3 (~35% extra products, estimate).
- Ranked: fuse spread with the first forward step and the pointwise with the first inverse step
  (one L2 pass each); three variables per step for size 2.
- Split: `forward_bottom` is ~20% of max_random (intrinsics, per-lane twiddle loads); the
  arena's huge pages still cost ~0.3 ms of zeroing (estimate).
