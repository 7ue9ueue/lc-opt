# multivariate_convolution

K <= 18 variables, n_l >= 2, N = prod n_l <= 2^18; print f g mod (x_1^n_1, ..., x_K^n_K) mod
998244353. 10 s.

Record when opened: 117 ms (issue #36). Best judged: ours, 20 ms:
[409313](https://judge.yosupo.jp/submission/409313) (round 1 `main.cpp`).

Official tests (shapes): twos (18 and 17 variables of size 2), threes (14 and 13 variables of
sizes 2 and 3), max_random (7 variables of 2..10, N = 151200 and 181440), dim1 (N ~ 2^18), dim2,
small, k0.

## Design

`solution.cpp` picks one of two methods.

- Graded (any shape; used on max_random, dim1, dim2, small). chi(i) = sum_{j<k} floor(i / P_j),
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
- Output: `../fixed_width.hpp` (judge-specific padding). `.preinit_array` start and `_exit`.
- `-DFORCE_GRADED` forces the graded method; `-DBLOCK_BYTES=1` makes every non-lane variable a
  top one. `stress.py` runs both and the default build against `brute.cpp`.

Sources: the ranked transform is the subset-convolution technique of Björklund, Husfeldt, Kaski,
Koivisto, "Fourier meets Möbius: fast subset convolution", STOC 2007 (recalled). The chi grading
for truncated multivariate products is a known technique (recalled from competitive-programming
folklore; no code read). Montgomery reduction with `vpmuludq`, `vpermd` lookups: standard.

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

## Next

- max_random is bound by 7 lib convolutions of 2^19 (58% and 69% filled). With a transform API
  that exposes forward, pointwise and inverse separately, split the top variable into q pieces
  (schoolbook in y, grades for the rest): max_random_00 at q = 10 needs 30 transforms of 2^15
  per grade instead of 3 of 2^19 (-37% butterflies; guess). `lib/ntt` only offers the fused
  product, so this needs a transform here or a lib change (owned by #33).
- Ranked: prune rank ranges in the block transforms (zero ranks per vector are known), radix-4
  passes with lazy reduction, tighter pointwise loops (per-lane caps differ by up to 3).
