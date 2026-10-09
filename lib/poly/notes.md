# lib/poly

Power series modulo P = 998244353 for `problems/polynomial/` (issue #95). Two layers:

- `transform.hpp`: transforms of length 2^6 .. 2^25 and products in the transform domain.
- `calculus.hpp`: coefficient-wise operations: `derivative`, `divide_by_index` (integration).
- One header per operation on top: `inverse.hpp`, `exp.hpp` (Newton iterations), `log.hpp`
  (division f'/f).

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
- Separate calls instead of lib/ntt's fused convolution: `forward`, `inverse` (in place or out
  of place), `multiply`, `multiply_add`, and fused depth-first passes: `cyclic_product` (forward
  of a, leaf products with a stored transform b, inverse), `inverse_product` (leaf products of
  two stored transforms, inverse), `inverse_product_sum` (the same for a sum of up to 3
  products; each leaf product reduced, then added), `forward_product` (forward of a, leaf
  products with b, kept as a transform).
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

## Calculus

- `derivative`: one Montgomery product per vector with the indices k 2^32 mod P, kept as a
  vector and stepped by 8 2^32 mod P.
- `divide_by_index(a, first, q)`: q[i] = a[i] / (first + i), Montgomery's batch inversion in 4
  interleaved vector chains (32 lanes; lane l takes the integers first + l + 32 j). Forward:
  prefix products into q; one scalar inversion for all 32 lane totals; backward: each reciprocal
  from the previous prefix and the running inverse, times a[i]. The factor 2^-32 of each
  Montgomery step cancels between the passes. 4 Montgomery products per coefficient; 0.43 ms
  for the ~2^19 divisions of exp at N = 500000 (`lc-amd`). `detail::divide_by_index` takes a
  loader for a, so callers fuse the computation of a into the backward pass.
- Integral: `divide_by_index(a, 1, q.subspan(1))`, q[0] = 0.

## Exp

Newton from m to 2m, with g = exp(f) mod x^m, h = 1 / g mod x^(m/2) and H = T_m(h) (the previous
step's transform of length 2m' = m), d = f', q = d mod x^(m-1):
- G_lo = T_m(g); e = (g h)[m/2, m) by `inverse_product` of G_lo and H, upper half;
  h[m/2, m) = -(x^(m/2) e h mod (x^m - 1))[m/2, m) by `cyclic_product` with H.
- r = x q g mod (x^m - 1): T_m(r) = `forward_product` of x q with G_lo, then r by an
  out-of-place `inverse`. As polynomials of degree < m, r = (g q - g') / x^(m-1) + x g', since
  g' = g q mod x^(m-1).
- H = T_2m(h); T_2m(r) = [T_m(r), `forward_upper` of r]; t = h r mod x^m by `inverse_product`,
  lower half. Then t exceeds h (g q - g') / x^(m-1) mod x^m by x h g' = x q mod x^m.
- log g = integral of g'/g, g'/g = q - h (g q - g') mod x^(2m-1), so
  s = (f - log g)[m, 2m) = (d[m-1+i] + t_i - d[i-1]) / (m + i): `divide_by_index` with a loader.
- g[m, 2m) = g s mod x^m: G = T_2m(g) by `forward_upper`, `cyclic_product` of x^m s (in place at
  w[m, 2m)), upper half.
- Per step 8 transforms of length 2m (in halves: T_m(g) 1, e 1, h 2, T_m(r) 1, r 1, T_2m(h) 2,
  upper half of T_2m(r) 1, t 2, upper half of T_2m(g) 1, g s 4) and 3.5 leaf products of that
  length. In all about 16 T(n) + 7 LP(n) for n = 2^19 (inverse: 10 T + 4 LP).
- Below 64 coefficients: the recurrence n g_n = sum_k k f_k g_(n-k); h mod x^32 by
  `inverse_direct`.
- f is only read by the first pass (into d = f'), so g may be f.
- Scratch: d, G, H, w (length 2^lg each), h (2^(lg-1)).

## Log

log f = integral of q, q = f'/f mod x^(n-1), d = f'. Blocked division: B blocks q_j of k
coefficients (k the least power of two >= (n - 1) / 4, at least 32; so B is 2 to 4), from
h = 1 / f mod x^k (`inverse`), with transforms of length 2k. With Q = q mod x^(jk), d - f Q is
divisible by x^(jk), and
- q_j = h (d - f Q)[jk, (j+1)k) mod x^k: `cyclic_product` of the residual r_j with H = T(h),
  lower half.
- (f Q)[jk, (j+1)k) = sum_(i<j) (W_(j-i) q_i)[k, 2k), W_t = f[(t-1)k, (t+1)k): a middle product,
  exact in the upper half of the cyclic product (degree < 3k - 1). One `inverse_product_sum`
  of the stored T(W_t) and T(q_i), upper half; d subtracted (computed on the fly from f).
- Cost with U = a transform of length 2k and V = a leaf product of that length: inverse to k
  5 U + 2 V, T(h) U, T(W_t) (B - 1) U, per block 2 U + V (q_j), U (T(q_j), not for the last)
  and U + j V (residual): (3 + 5 B) U + (2 + B + B (B - 1) / 2) V. B = 2 is Karp and
  Markstein's division (h to n/2, 13 U + 5 V). For n - 1 = 499999: B = 4 at k = 2^17 gives
  23 U + 12 V, i.e. 11.5 T(2^19) + 6 LP(2^19), against 13 T + 5 LP for B = 2 at k = 2^18;
  B = 8 at 2^16 would be 10.75 T + 9.5 LP (worse with LP ~0.7 T).
- In place: g may be f. f is read by the inverse and the forwards of W_t first; block j reads
  f[jk + 1, (j+1)k + 1) (d on the fly) before it writes g[jk + 1, (j+1)k + 1).
- Scratch: 2 B buffers of length 2k (H, W_1 .. W_(B-1), q_0 .. q_(B-2), work); the inverse's
  scratch overlaps them. The integral divides by index with `divide_by_index`, whose loader
  reads q_j.
- exp's step computes log g [m, 2m) for its own g (degree < m, with h = 1/g kept from the
  previous step), not log of a given f, so the two share only calculus.hpp.

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

2026-10-09, claude (issue #63, exp_of_formal_power_series):
- Added `calculus.hpp` (derivative, divide_by_index) and `exp.hpp`; `transform.hpp` and
  `inverse.hpp` unchanged. Tests: derivative (in place, short and long f), divide_by_index
  (sizes 0 .. 100 and to 10^5, first up to P - n) against scalar code; exp against the O(n^2)
  recurrence (n <= 160 and edge cases), longer outputs by a prefix and i g_i = sum k f_k g_(i-k)
  at random i, sizes 2^k - 1, 2^k, 2^k + 1 to 2^20, in place and not.
- exp at N = 500000 on `lc-amd`: 14.1 ms in process (phases in
  problems/polynomial/exp_of_formal_power_series/notes.md). Of that ~10.7 ms in transform
  levels and ~3.2 ms in leaf products (estimate from the per-op times above).
- e = g h from the stored T_m(g) (copy, `multiply`, `inverse`) instead of `cyclic_product`
  (which transforms g again), the uncorrected r, vector negation: 14.15 vs 14.70 ms.
- Added `Transform::inverse_product`, `forward_product` and an out-of-place `inverse` (new
  bottoms; `ProductBottom` unchanged). exp with them: 13.55 vs 14.26 ms (0.950); inv's `.text`
  is byte-identical. Tests: each against the cyclic product, output halves, in place on either
  operand, forward_product from x^shift in; a mutation (no 2^32 correction) fails them.
- `vbroadcastss` (`_mm256_broadcast_ss`) for the broadcasts of b in `leaf_product`
  (-1 cycle in isolation, see above): `judge.py bench` 21 rounds, exp 1.0003, inv 0.9984.
  Not kept.
- Considered, not done: a relaxed (online) exp with
  B-ary blocks: with leaf products at ~0.75 of a transform, 16-ary blocks cost ~4 levels x (15
  LP + 4 T) per coefficient, far above Newton; it would need full-depth transforms with
  cheap pointwise products, and its serial base case (one modular chain per coefficient,
  ~20 cycles) alone costs ~3 ms.

2026-10-09, claude (issue #64, log_of_formal_power_series):
- Added `log.hpp` (`log`, `log_log`, `log_scratch`); `transform.hpp`, `inverse.hpp`,
  `calculus.hpp` and `exp.hpp` unchanged. Tests: log against the O(n^2) recurrence (n <= 160,
  edge cases f = 1, f = 1 - x, f shorter and longer than n), longer outputs by a prefix and
  i f_i = sum k g_k f_(i-k) at random i and around the split m; sizes 2^k - 1 .. 2^k + 2 to
  2^20 + 1.
- log at N = 500000 on `lc-amd`: 10.81 ms in process (phases in
  problems/polynomial/log_of_formal_power_series/notes.md): inverse to 2^18 3.89, three
  `cyclic_product`s 5.09, two forwards 1.28, calculus 0.55.
- Blocked division (4 blocks, h to 2^17; see Log above) with a new `inverse_product_sum`:
  10.24 vs 10.81 ms in process. `InverseProductBottom` took K pairs; `inverse_product` was
  K = 1. inv's `.text` was byte-identical; exp's code moved (same size), `judge.py bench` 21
  rounds 0.9994 on `lc-amd`. Tests: `inverse_product_sum` of 1, 2, 3 pairs against cyclic products, output
  halves, into an operand; log in place; sizes at the block boundaries (n - 1 = 3k, 3k + 1).
  The extra leaf products of a sum cost 0.27-0.29 ms each at 2^18 (the first, with the
  inverse, 0.51 ms in all). Merged as #128.
- #128's CI timed exp 1.0802 on an Intel Xeon 6973P-C (AMD machines 0.9944, 1.0007); on
  `lc-intel` (-march=native, AVX-512 code) `judge.py bench` 21 rounds: 1.0716, so real. Cause:
  the template `InverseProductBottom<K>` with K = 1 compiled exp's products differently.
  Fix: the single-product `InverseProductBottom` restored as before #128;
  `InverseProductSumBottom<K>` (K = 2, 3) holds K of them and reuses their window fills. exp's
  and inv's `.text` are again identical to before #128 on `lc-amd` and `lc-intel`. log
  (`judge.py bench`, 21 rounds, ratios to Karp-Markstein): `lc-amd` #128 0.9600, fix 0.9582;
  `lc-intel` 0.9648, 0.9544.

## Sources

- lib/ntt (our refactor of QPoly): table layout, kernels, recursion.
- D. Harvey, "Faster algorithms for the square root and reciprocal of power series",
  Math. Comp. 80 (2011), https://arxiv.org/abs/0910.1926 (read for the cost analysis; no code).
- Middle product and Newton with transform reuse: G. Hanrot, M. Quercia, P. Zimmermann, "The
  middle product algorithm I", AAECC 14 (2004) (as cited by Harvey).
- Exp by Newton iteration with a simultaneous inverse: the standard scheme (Brent; Hanrot and
  Zimmermann, "Newton iteration revisited", 2004; D. Harvey, "Faster exponentials of power
  series", https://arxiv.org/abs/0911.3110, for the cost accounting). Derived and written here;
  no code read.
- Batch inversion: P. Montgomery, "Speeding the Pollard and elliptic curve methods of
  factorization", Math. Comp. 48 (1987).
- Division with the inverse's last Newton step merged: A. Karp, P. Markstein, "High-precision
  division and square root", ACM TOMS 23 (1997) (the idea, as described by Hanrot and
  Zimmermann above). The blocked form (residuals by middle products of stored transforms) is
  the usual blockwise division; derived and written here, no code read.
