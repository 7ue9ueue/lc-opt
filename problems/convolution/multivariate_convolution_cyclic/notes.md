# multivariate_convolution_cyclic

Prime p <= 10^9, K <= 18 axes, n_i >= 2, n_i | p - 1, N = prod n_i <= 2^18. Print f g mod
(x_i^n_i - 1) mod p. 10 s.
Record when opened (issue #37): 117 ms. Best judged: none yet.
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
  primes (`transform.hpp`, `kernels.hpp`: convolution_mod_1000000007's run-time-modulus copy of
  lib/ntt), CRT straight to residues mod p (Montgomery, p odd), fold j + D_r onto j.
  Coefficients < 2^18 p^2 < 2^78; the primes' product is 2^89.6.
- Split: axes above 48 are long (the shortest dropped while the transform would exceed 2^20);
  a shorter axis joins when a cost model says so (13 ns per transform word, 0.5 ns per element
  and axis length).
- Pointwise product when no axis is long. The scale 1 / prod(short n) is folded into the CRT
  constants. Output `fields.hpp` (convolution_mod's, 10-byte fields). `.preinit_array` start.

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
- Next: the transforms are 70% of the dim cases (7 ms of 3 x 2^19). Short path: fuse adjacent
  axes into one pass, split short axes into coprime factors (10 = 2 x 5: 7 terms instead of 10).

## Sources

- Chinese remainder theorem for cyclic groups (Z_ab = Z_a x Z_b for coprime a, b), as in the
  Good-Thomas prime-factor FFT; invariant-factor decomposition of finite abelian groups.
- Kronecker substitution for multivariate products.
- Our own code: `../convolution_mod_1000000007` (transform, CRT), `../convolution_mod`
  (`fields.hpp`), `lib/io`.
