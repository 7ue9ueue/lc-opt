# multivariate_convolution_cyclic

Prime p <= 10^9, K <= 18 axes, n_i >= 2, n_i | p - 1, N = prod n_i <= 2^18. Print f g mod
(x_i^n_i - 1) mod p. 10 s.
Record when opened (issue #37): 117 ms. Best judged: ours, 12 ms, no spike:
[409321](https://judge.yosupo.jp/submission/409321) (`main.cpp` of #150). Earlier: 15 ms,
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
  primes (`lib/multimod`, `Bounded` input), CRT straight to residues mod p (Montgomery, p odd), fold j + D_r onto j.
  Coefficients < 2^18 p^2 < 2^78; the primes' product is 2^89.6.
- One cyclic factor (all long tests): the places are [0, D), so a and b need no clearing and the
  CRT folds as it reads (c_j + c_{j+D}). One long axis of stride 1 (dim1, dim2_00): the
  transforms read f and g in place (masked loads, never past the count) and the CRT writes into
  f. Two long axes (dim2_01, dim2_02): place j = r n_c + i_c with i_q = j mod n_q, so gather and
  scatter run over blocks of 16 values of i_c, by place inside a block (f's rows stay in L1,
  places come in runs of 16).
- The transform's first radix-4 level reads the input directly (no copy pass) when sparse.
- Split: axes above 48 are long (the shortest dropped while the transform would exceed 2^20);
  a shorter axis joins when a cost model says so (13 ns per transform word, 0.5 ns per element
  and axis length).
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

## Sources

- Chinese remainder theorem for cyclic groups (Z_ab = Z_a x Z_b for coprime a, b), as in the
  Good-Thomas prime-factor FFT; invariant-factor decomposition of finite abelian groups.
- Kronecker substitution for multivariate products.
- Our own code: `../convolution_mod_1000000007` (transform, CRT), `../convolution_mod`
  (`fields.hpp`), `lib/io`.
