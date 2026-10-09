# lib/poly

Power series modulo P = 998244353 for `problems/polynomial/` (issue #95). Two layers:

- `transform.hpp`: transforms of length 2^6 .. 2^25 and products in the transform domain.
- One header per operation on top: `inverse.hpp` (Newton iteration).

APIs and usage: the header of each file. Tests: `test.cpp` (O(n^2) references; sizes 1..64,
powers of two and their neighbours up to 2^20, random sizes; also run under ASan/UBSan in CI).

## Transform layer

- The transform of a of length n is n/8 leaves: leaf p = a mod (x^8 - w_p), 8 canonical
  coefficients, w_p = (-1)^p r[p >> 1] with lib/ntt's twiddle table r. w_p does not depend on n:
  the transform of length n is the first half of that of length 2n (`forward_upper` gives the
  second half, "doubling").
- Same tree as lib/ntt: radix-4 groups, depth first, subtrees of 256 vectors level by level; lib/ntt's
  asm kernels for every level with h >= 4 (`forward`, `forward_identity`, `inverse`,
  `inverse_identity`, `scale_radix2`) and its inverse radix-4 top level. New here, intrinsics:
  the bottom level (h = 1) with or without leaf products, the forward top level, the inverse top
  level for one output half.
- Separate calls instead of lib/ntt's fused convolution: `forward`, `inverse`, `multiply`,
  `multiply_add`, and `cyclic_product` (forward of a, leaf products with a stored transform b,
  inverse, depth first in one pass).
- Sources: the forward top level reads x^shift in[0, size) from any span (in place or not);
  coefficients outside are zero and not read. Outputs: `Half` computes one half only.
- Leaf product: a window [w a, a] (canonical) gives x^i a mod (x^8 - w) as words [8 - i, 16 - i);
  16 `vpmuludq` into 64-bit sums (8 products < P^2 plus the Montgomery term stay below 2^64), one
  Montgomery reduction per lane. The factor 2^-32 is undone by the inverse's scale in
  `cyclic_product`, and by one Shoup multiplication by 2^32 in `multiply`.
- Memory: `Arena`, one mapping in transparent huge pages, spans 32-byte aligned with 64 bytes after
  each (the forward kernels read 4 bytes past the end).

## Inverse

Newton from k to 2k with transforms of length 2k: G = T(g_k) (lower half input);
e = f g_k mod (x^2k - 1), upper half only; g[k, 2k) = -(x^k e[k, 2k) g_k mod (x^2k - 1))[k, 2k),
upper-half input and output. 5 transforms of length 2k and 2 leaf products per step, about
10 T(n) + 4 LP(n) in all. Below 32 coefficients: the direct recurrence.

## Measurements

AMD EPYC 7B13 (`lc-amd`, 3.48 GHz), GCC 15.2, judge flags, 2026-10-09. ns per coefficient,
minimum over repeats, data in cache (`probe/sizes.cpp`, scratch):

| lg | forward | inverse | cyclic_product | multiply |
|---|---|---|---|---|
| 10 | 0.58 | 0.63 | 2.18 | 1.28 |
| 14 | 0.83 | 0.79 | 2.56 | 1.27 |
| 18 | 1.12 | 1.05 | 3.13 | 1.28 |
| 19 | 1.17 | 1.17 | 3.28 | 1.30 |
| 20 | 1.27 | 1.19 | 3.42 | 1.29 |

`inverse` of 500000 coefficients: 8.26 ms in process (last step at 2^19: forward 0.64,
products 1.77 and 1.69).

## Log

2026-10-09, claude (issue #62, inv_of_formal_power_series):
- Built the layer above. Checks: `test.cpp` PASS at -O2 and ASan/UBSan, `-march=x86-64-v3` and
  `-march=native` (`lc-intel`, AVX-512 host).
- Kernel costs on `lc-amd` (scratch probes): `vpmuludq` and `vpmulld` 2 per cycle, but 6
  `vpmuludq` + 6 `vpaddq` take 4.3 cycles (they share pipes). A leaf product from stored windows
  takes 17 cycles (minimum ~11 by uop count); without broadcasts and reduction still 13.8. Two
  leaves interleaved 18.9, two accumulators per parity 17.0, windows by `vpermd` 56.6.
  `vbroadcastss` instead of `vpbroadcastd`: -1 cycle in isolation (not adopted, intrinsics).
- Store forwarding: windows written by two 32-byte stores and read by unaligned loads: 29.5
  cycles per leaf when read one leaf later, 24.5 when three later, 18 when stored long before.
  In `cyclic_product` the butterflies between them hide most of it.
- Factor form (b stored as windows [w b, b], 2n words, built by its forward transform; products
  broadcast a's words from a small buffer): `cyclic_product` 3.24 vs 3.26 ns per coefficient at
  2^19, its forward +0.25. Not kept.
- lib/ntt's asm `bottom_last` (leaf products + inverse butterfly) inside `cyclic_product`: 2.13 vs
  2.06 (intrinsics) at 2^10, 3.28 vs 3.26 at 2^19. Not kept.
- Tile size 64, 256, 1024, 4096 vectors: within 1% at 2^12..2^19. Kept 256.
- Top level: out-of-place 60 us at 2^19 (write-allocate traffic), in place 34 us, lib/ntt's
  `forward_identity` 44 us. Sources are kept for the API (no copies or zero fills by callers);
  the program gained ~0.5%.
- Plain transforms instead of Montgomery form (2^32 in the scale): forward 1.27 -> 1.17 ns at 2^19.
- Algorithms considered: Harvey's 13/9 M(n) reciprocal (blocked third-order Newton) needs
  ~4.5 s^2 leaf products per 3s blocks; with a leaf product at ~0.47 of a transform per
  coefficient, the estimate is 12.5 T(n) (s = 2) and 13.6 T(n) (s = 3) against 11.9 T(n) for the
  Newton above. Schoenhage's 1.5 M(n) step needs length-3k transforms (3 does not divide P - 1).
  Neither pursued.

## Sources

- lib/ntt (our refactor of QPoly): table layout, kernels, recursion.
- D. Harvey, "Faster algorithms for the square root and reciprocal of power series",
  Math. Comp. 80 (2011), https://arxiv.org/abs/0910.1926 (read for the cost analysis; no code).
- Middle product and Newton with transform reuse: G. Hanrot, M. Quercia, P. Zimmermann, "The
  middle product algorithm I", AAECC 14 (2004) (as cited by Harvey).
