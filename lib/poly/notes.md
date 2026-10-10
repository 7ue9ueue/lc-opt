# lib/poly

Power series modulo P = 998244353 for `problems/polynomial/` (issue #95). Two layers:

- `transform.hpp`: transforms of length 2^6 .. 2^25 and products in the transform domain.
- `calculus.hpp`: coefficient-wise operations: `derivative`, `divide_by_index` (integration).
- One header per operation on top: `inverse.hpp`, `exp.hpp` (Newton iterations), `log.hpp`
  (division f'/f), `pow.hpp` (c (f / f[0])^e as exp(e log)), `sqrt.hpp` (Newton iteration).
- `sparse.hpp`: linear recurrences with few taps, for series with few nonzero terms (issues
  #69-#73); independent of the transform layer.
- `composition.hpp`: f(g) mod x^n by Kinoshita and Li's algorithm (issue #67); its own bottoms
  and tables on top of the transform layer.

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
  coefficients outside are zero and not read. Outputs: `Half` computes one half only;
  `cyclic_product` multiplies its output by a constant c at no cost (c is folded into the
  inverse's final scale), so a negated product needs no pass of its own.
- Leaf product: a window [w a, a] (words <= P) gives x^i a mod (x^8 - w) as words [8 - i, 16 - i);
  16 `vpmuludq` into 64-bit sums (8 products < P^2 plus the Montgomery term stay below 2^64), one
  Montgomery reduction per lane. The factor 2^-32 is undone by the inverse's scale in
  `cyclic_product`, and by one Shoup multiplication by 2^32 in `multiply` and `forward_product`.
  Inline asm in a fixed order (step i: broadcast b[i], two multiplies, the adds of step i - 1).
  `fill_windows` writes a group's 4 windows: weights y, -y, z, -z with y, z broadcast from the
  table, -(y a) as 2P - y a. The product bottoms are always inlined and unrolled.
- Memory: `Arena`, one mapping in transparent huge pages, spans 32-byte aligned with 64 bytes after
  each (the forward kernels read 4 bytes past the end).

## Inverse

Newton from k to 2k with transforms of length 2k: G = T(g_k) (lower half input);
e = f g_k mod (x^2k - 1), upper half only; g[k, 2k) = -(x^k e[k, 2k) g_k mod (x^2k - 1))[k, 2k),
upper-half input and output, the sign in the product's scale. 5 transforms of length 2k and 2
leaf products per step, about 10 T(n) + 4 LP(n) in all. Below 32 coefficients: the direct
recurrence. `inverse_step` is one step; f may start at its work buffer (f is overwritten) and
its output may be the work buffer's upper half, so a caller can run the last step in f's buffer
(inv_of_formal_power_series does: 3 huge pages instead of 5 at N = 500000).

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
- Code: `exp` = `derivative` into d, `exp_direct`, then `detail::exp_newton` (the steps, from any
  d and any g[0] != 0; `power` reuses it).

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
- Code: `detail::log_derivative(t, f, n, scratch, sink)` computes the blocks of q for any
  f[0] != 0 and hands each to sink(first, q); `log` integrates them in its sink, `power` scales
  them into its d.
- exp's step computes log g [m, 2m) for its own g (degree < m, with h = 1/g kept from the
  previous step), not log of a given f, so the two share only calculus.hpp.

## Power

g = c (f / f[0])^e = c exp(e log(f / f[0])) mod x^n for residues e, c and f[0] != 0. For an
integer M >= 0 and n <= P, f^M = power(f, M mod P, f[0]^M): (f / f[0])^M has constant term 1,
so only M mod P matters. Callers handle leading zeros (f = x^k u gives x^(kM) u^M). A square
root is power(f, 1/2, sqrt(f[0])).
- g solves g' = d g with g[0] = c and d = e f'/f. `detail::log_derivative` (log's blocked
  division, any f[0] != 0) hands q = f'/f to a sink block by block; the sink stores e q into d.
  Then `exp_direct` (times c) gives g mod x^64 and `detail::exp_newton` (exp's steps) the rest.
  Every exp step is invariant under scaling g (h = 1 / g and g'/g follow), so g[0] = c costs
  nothing.
- Against log then exp, this skips the integral of q (`divide_by_index`), exp's derivative and
  the passes f / f[0], times e, times c.
- In place: log_derivative reads all of f before g is written.
- Scratch: d (2^exp_log(n) words), then the larger of log_derivative's 2B buffers and
  exp_newton's 3.5 buffers of length 2^exp_log(n). log_derivative_scratch counts at least 3
  buffers: for B = 1 (n <= 33, only power) the inverse's scratch follows T(h).
- In process at N = 500000 (`lc-amd`, warm, medians of 11): log_derivative 9.93 ms (blocks
  4.32, 1.70, 1.97, 1.93; block 0 includes the inverse, T(h) and the forwards of W_t),
  exp_newton 13.34 ms; first use +0.6 ms (page faults).
- Alternatives considered, not tried:
  - Newton directly on the ODE f g' = e f' g: residual A = e f' g - f g' from fresh
    transforms of f, x f', g and x g' (length 2m), then A / (f g) by two products with 1/f and
    1/g, then g s. About 14.5 transforms of length 2m per step plus 1/f to n/2: ~34 T(n)
    against 27.5 T(n) for log (11.5) and exp (16).
  - exp's T_m(x q) at m = 2^17, 2^18 from the log's stored T(q_0), T(q_1) by leaf-wise shifts
    (x^k is a scalar per leaf): saves ~0.4 ms of forwards, keeps 2 MB more live.

## Sqrt

g = sqrt(f) mod x^n with g[0] = c, c^2 = f[0] != 0. Newton from m to 2m keeps h = -1 / g mod
x^(m/2) and H = T_m(h) (the previous step's transform of length 2m' = m). The sign of h saves
negations: the inverse update and the root step both come out with the right sign.
- G = T_m(g). e = (g h)[m/2, m) by `inverse_product` of G and H, upper half; h[m/2, m) =
  (x^(m/2) e h mod (x^m - 1))[m/2, m) by `cyclic_product` with H. Now h = -1 / g mod x^m.
- r = (g^2 - f)[m, 2m) / 2: g^2 mod (x^m - 1) by `inverse_product` of G with itself equals
  f[0, m) + (g^2)[m, 2m), since g^2 = f mod x^m and deg g^2 < 2m - 1. One pass subtracts f[0, m)
  and f[m, 2m) and halves.
- g[m, 2m) = h r mod x^m: T_2m(h) by `forward` (also the next step's H), `cyclic_product` of r,
  lower half. (g + x^m d)^2 = f mod x^2m needs 2 g d = (f - g^2)[m, 2m) mod x^m, so d = h r.
- Per step 11 transforms of length m (G 1, e 1, h 2, g^2 1, T_2m(h) 2, d 4) and 5 leaf products
  of that length.
- Last step, m < n <= 2m, transforms of length m only. With rest = n - m: if rest <= m/2,
  d = h r mod x^rest by one `cyclic_product` with H (h is known to m/2): 4 transforms. Else
  Karp and Markstein's split: d0 = h r0 mod x^(m/2) (r0 = r mod x^(m/2)); g d = -r gives
  g d1 = -(r + g d0)[m/2, rest) mod x^(rest - m/2), so d1 = h (r + g d0)[m/2, rest), with
  (g d0)[m/2, m) exact in the upper half of the cyclic product of g (degree < m) and d0
  (degree < m/2). 8 transforms of length m (G 1, g^2 1, three cyclic products 6), 4 leaf
  products. Against a full step (11 transforms, h to precision m, d at length 2m) this keeps
  the largest transform at m: 2^18 for n = 500000.
- In all, for n = 500000 (m = 2^18 last): ~9.5 T(2^19) and 4.5 LP(2^19); the inverse of
  `inverse.hpp` takes 10 T(2^19) and 4 LP(2^19).
- Below 64 coefficients: 2 c g_i = f_i - sum_(0<j<i) g_j g_(i-j); h mod x^32 by `inverse_direct`
  of g, negated.
- Scratch: G, T(h) and work (length m_last each), h (m_last / 2); in the last step h holds
  r[m/2, rest) (H keeps h's transform). f and g must not overlap: every step reads f[0, 2m).
- Alternatives considered, not done:
  - power(f, 1/2, c) (exp of log / 2): 2.25 times slower whole process (see Log).
  - h at full precision m (the update after g[m, 2m)): 17 transforms per step.
  - The root step at length m with h at m/2 (the last step's split) in every step: 6
    transforms for d, but h still needs its update (3) and T_2m(h) (2): 13 per step.
  - Blocks of s coefficients with transforms of length 2s for the last stage (as log's
    division: windows of g against stored T(d_i)): for n = 500000 and s = 2^16, about 21
    transforms of length 2^17 and 10 leaf products, against the split's 8 of length 2^18
    (~16 of 2^17) and 4 (8 of 2^17). Newton to 2^17, then blocks of 2^16: 23 transforms of
    2^17 and 33 leaf products against 27 and 13; a leaf product at 2^17 costs ~0.87 of a
    transform (0.12 vs 0.138 ms, `lc-amd`).
  - The inverse square root f^(-1/2) is 1/g = -h already; its own Newton step, u + u (1 -
    f u^2) / 2, needs f u^2 mod x^2m, a longer product than g h.
  - Harvey's 4/3 M(n) square root (Harvey 2011 below; 8 T(n) with M(n) = 6 T(n)): blocked,
    more leaf products per transform; not tried (as the 13/9 reciprocal, issue #62 below).

## Sparse

`Recurrence`: g[i] = r[i] + sum over taps (d, c) of c g[i - d], at most 16 taps, r sparse;
`next(out, count)` produces the next coefficients into any window that holds the history before
it (a ring, or one array). For 1/f: taps (i_k, -a_k / a_0), r = [1 / a_0].
- Blocks of 16. Short taps (d < 16): the state is the last w = max d values;
  g[n + t] = sum_(j < w) A[t][j] g[n - 1 - j] + sum_(s <= t) u[t - s] r'[n + s], u the short
  taps' impulse response, r' = r plus the long taps' terms (16-wide loads at n - d, all before
  the block). A and u are built once by running the short recurrence on unit states.
- Where no long tap reaches and r is zero (all of a dense f's output): 64 coefficients per step
  from one state, in four independent blocks of 16, the block holding the next state first.
  w products per coefficient and no dependency inside a step. One kernel per w (1..15), fully
  unrolled.
- Products: coefficient times 2^32 (Montgomery form) and value, both below P, by `vpmuludq`
  into 64-bit sums laid out as qwords of the even and odd outputs (no shuffles before the
  products). Reduction per sum: up to 12 products, one Montgomery step (< 4P) and two
  subtractions; up to 17 (any sum below 2^64), first 2^32 h + l -> h (2^32 mod P) + l < 2^62.
- Blocks with long taps or r: r' reduced, the A part reduced, then h 2^32 + U r' (17 products).
  Without short taps the block is r' itself.
- Costs (`lc-amd`, in memory): about 0.125 + 0.05 w ns per coefficient for w <= 12 (0.463 at w = 7);
  each column is 4 `vpmuludq` + 4 `vpaddq` per 16 values, 2.8 cycles.
- Next, for exp, log, pow and sqrt (#70-#73): their recurrences have coefficients linear in n
  (log's n g_n is constant-coefficient: G = n g, then a division by n); dense small-tap inputs
  then need a different block step, and bulk inverses of 1..N (`calculus.hpp` has batch
  inversion).

## Composition

h = f(g) mod x^n, g[0] = 0, as the transpose of power projection (Kinoshita and Li, Sources).
m = 2^T >= max(n, 128); n <= 32 by Horner's rule.
- Levels: Q_0 = 1 - y g(x), Q_(s+1)(x^2, y) = Q_s(x, y) Q_s(-x, y) mod x^L, L = m / 2^s, Y = 2^s
  the y degree; Q_s(x, 0) = Q_s(0, y) = 1. Power projection maps w to ([x^(m-1)] w g^i)_i by
  P_(s+1) = odd part of P_s Q_s(-x); composition applies its transpose to f: P_s from P_(s+1)
  by a middle product with Q_s(-x), and h = P_0 reversed.
- Layout: x = z, y = z^(2L) (Kronecker), transforms of length 4m: a product carries nothing from
  x into y while the x degree stays below 2L and wraps y mod y^(2Y). Wraps that land on a known
  row (row 0 of Q_s(x) Q_s(-x) is 1) are undone exactly, so 2Y rows suffice for degree 2Y. x
  must be the low part: z -> -z then maps Q(x, y) to Q(-x, y), and in the leaf domain Q(-z) mod
  (z^8 - w) is the leaf with its odd coefficients negated.
- Forward pass (`LevelBottom`, the bottom of the transform of Q_s at 4m): canonical leaves are
  the stored level; leaves 2p, 2p + 1 (moduli z^8 -+ s, s = r[p], s^2 = w_p) give V =
  Q_s(x) Q_s(-x) mod (u^8 - w_p), u = z^2, as the CRT of a(z) a(-z) = e(u)^2 - u o(u)^2 mod
  (u^4 - s) and the same for b: 20 products per leaf instead of a 64-product leaf product, 8
  pairs per step in transposed form (one pair per lane). Then the inverse of V at 2m and the
  next layout (truncate x, unwrap row 2Y).
- Backward pass: with P_(s+1) reversed in x and y (layout stride L), the middle product is the
  plain product R = P(z^2) Q_s(-z) mod (z^4m - 1); rows Y .. 2Y - 1 of R, x below L, are P_s
  reversed, the next input at once (the reversals cancel). `CompositionBottom`: P(z^2) mod
  (z^8 -+ s) = lo +- s hi from P's leaf p at 2m, 4 terms, so leaf products of 4 by 8; 8 pairs per
  step transposed, s folded into the wrapped operands. Per generic level: transform of P at 2m,
  inverse at 4m with this bottom (upper half).
- One-dimensional levels: 0 (Graeffe of g at 2m; h = p1(x^2) - p0(x^2) g(-x) with the same
  bottom at 2m, lower half), T - 1 (Q = 1 + x q(y): one cyclic product of length m) and T - 2
  (Q = 1 + x q1 + x^2 q2 + x^3 q3: four products of length m/2; q of level T - 1 = 2 q2 - q1^2).
- Costs at m = 8192 (`lc-amd`, in process, µs): per generic level forward 48 (plain transform
  of 4m: 27) + 13 (inverse of 2m), backward 13 + 50 (plain inverse of 4m: 27); compose 1460.
- Memory: level s keeps its transform of 4m words (levels 0, T - 2, T - 1 less); 2 work spans of
  4m. compose_scratch(8000) = 504200 words.
- For power projection (#68, #86, #87): the forward pass is the same; P_(s+1) = odd part of
  P_s Q_s(-x) is, per leaf pair, the CRT of the odd parts of a_P(z) a_Q(-z) and b_P(z) b_Q(-z),
  so the same pair structure fits a forward-only pass with P's transform at 4m per level.
- `ntt::detail::multiply` (`times`) takes one factor for all lanes (the odd lanes reuse the even
  lanes' quotient); per-lane factors need `times_lanes`.

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

2026-10-09, claude (issue #65, pow_of_formal_power_series):
- Added `pow.hpp` (`power`, `power_log`, `power_scratch`) as a composition of `log` and
  `exp`; no other header changed (inv, exp and log bundles unchanged). Tests: power against
  the O(n^2) recurrence f[0] i g_i = sum_(0<j<=i) (e j - (i - j)) f_j g_(i-j) (n <= 160, e in
  {0, 1, 1/2, random}), longer outputs by a prefix and the recurrence at random i; edge cases
  f constant, f = 1 - x with e = -1, f shorter and longer than n, in place; sizes 2^k - 1 ..
  2^k + 1 to 2^20. A mutation (e + 1) fails them.
- pow at N = 500000 on `lc-amd`: 28.7 ms whole process (`judge.py test`); floor (read and
  write) 4.5 ms. Merged as #137.
- Fused power (see Power): `log.hpp` split into `detail::log_derivative` (always_inline, a sink
  per block) and the integral; `exp.hpp` into the derivative and `detail::exp_newton`
  (always_inline). `log_log` and `log_scratch` keep their old expressions (through
  `log_derivative_scratch` they changed log's code; bench 1.0034 on both VMs, then 1.0000).
  Code: inv's `.text` identical; exp's functions the same set and sizes except `exp`, from
  which `forward_upper` is no longer inlined (4038 -> 3023 instructions); log's `log`
  2435 -> 2387 instructions. Tests unchanged (power, log, exp pass, also ASan/UBSan).
- `judge.py bench` (ratios new/old): pow 0.9827 (`lc-amd`, 21 rounds), 0.9885 (`lc-intel`, 21);
  exp 0.9994, 0.9954 (21); log 1.0000, 0.9988 (41).

2026-10-09, claude (issue #66, sqrt_of_formal_power_series):
- Added `sqrt.hpp` (`sqrt`, `sqrt_log`, `sqrt_scratch`); no other header changed (inv, exp,
  log and pow bundles unchanged). Tests: sqrt against the O(n^2) recurrence (n <= 160, three
  kinds of f, c = either root), longer outputs by a prefix and g^2 = f at random coefficients;
  edge cases f = c^2, f = (1 - x)^2, f shorter and longer than n; sizes 2^k - 1 .. 2^k + 1,
  3 2^(k-2) and 3 2^(k-2) + 1 (the last step with one product or the split) to 2^20. A
  mutation (the split's residual without g d0) fails them. Fixture arena: 22 buffers of 2^20.
- In process at N = 500000 (`lc-amd`, scratch probe): 7.72 ms warm, 7.9 ms first use.
  Primitives (ms, min of 20): forward of a half-zero input at 2^17 / 2^18 0.138 / 0.303;
  `inverse_product` 0.260 / 0.531; `cyclic_product` (half-zero input, lower half) 0.387 / 0.821.
- Whole process (`judge.py bench`, `lc-amd`, 15 rounds, slowest 4): 12.87 ms against 28.91 ms
  for the same program with `power(u, 1/2, c)`.
- Merged as #148. Judged [409316](https://judge.yosupo.jp/submission/409316): AC 16 ms with a
  +9 ms launch spike; clean score 12 ms (record 25 ms).

2026-10-09, claude (issue #95, leaf products):
- Zen 3 costs (`lc-amd`, scratch probes, cycles per instruction, independent streams):
  `vpaddq`, `vpsubd`, `vpminud`, `vpblendd` 0.25; `vpmuludq`, `vpmulld`, `vpsrlq`, `vpalignr`,
  `vpshufd`, `vshufps` 0.5; `vperm2i128` 1.0; `vpbroadcastd` from memory uses no vector pipe.
  Multiplies and shifts use disjoint pipe pairs (6 + 6: 4 per cycle). Multiplies with adds: 4
  per cycle in the order M M A A, 3.0 in the order M A M A.
- Leaf product alone (64 leaves in L1, windows stored long before; cycles per leaf): GCC's code
  15.6, fixed-order asm 14.6, the same without memory operands 13.0, without the Montgomery step
  9.5 (16 multiplies, 14 adds). Each added op cost ~0.12 (shift), ~0.27 (add, blend), ~0.45
  (multiply): with half the ops multiplies, ~3.1 ops per cycle. Tree accumulation, two leaves
  interleaved, the Montgomery chain cut: within 0.5.
- In `cyclic_product` at 2^19 (ns per coefficient): 3.38; with the leaf product replaced by a
  copy 2.77, also without the window's w a 2.49. GCC compiled `ProductBottom::prepare` as a call
  (with `vzeroupper`), kept the loops over the 4 leaves rolled with stack round trips, and built
  the odd leaves' weights P - w in general registers (`vmovd`, `vpbroadcastd`).
- Changes: the product bottoms always inlined and unrolled; `fill_windows` for a group (y, z
  broadcast from the table, -(y a) as 2P - y a); `leaf_product` in inline asm (step i: broadcast
  b[i], two multiplies, the adds of step i - 1); unroll pragmas on `ForwardBottom`'s stores and in
  `InverseProductSumBottom`. API and transform format unchanged. Tests: `leaf_product` against
  scalar sums for windows with any words in [0, P] (all 0, all P - 1, all P, mixes) and canonical
  b; `fill_windows` against w a for groups 0 .. 999 and random; -O2 and ASan/UBSan,
  `-march=native` and `-march=x86-64-v3` (`lc-intel`). Mutations (no negation, a wrong window
  offset, no final reduction) fail them.
- In process at 2^19, ns per coefficient, medians of 40, same session, `lc-amd` (`lc-intel`),
  main -> this: `cyclic_product` 3.38 -> 3.07 (4.00 -> 3.92), `inverse_product` 2.14 -> 1.95
  (2.51 -> 2.47), `inverse_product_sum` of 2 3.13 -> 2.69 (3.93 -> 3.15), of 3 4.12 -> 3.46
  (4.47 -> 3.84), `forward_product` 2.35 -> 2.06 (2.67 -> 2.58), `forward` 1.24 -> 1.20
  (1.75 -> 1.74), `inverse` unchanged. Steps (`lc-amd`, `cyclic_product`): inlining and
  unrolling 3.19, asm leaf product 3.13, `fill_windows` 3.07. On `lc-intel` the first two steps
  made `inverse_product_sum` of 2 slower (3.85 -> 3.97); its unroll pragmas fixed that.
- Tried, not kept (`cyclic_product` at 2^19 unless noted):
  - Windows built in registers (`vperm2i128` + 6 `vpalignr` from a and w a): 31 vs 23 cycles per
    leaf in isolation (long chain from w a to the products; GCC spilled).
  - Windows prepared two groups ahead: no change (store forwarding is not the limit).
  - Four leaf products in one asm block on fixed registers: 1-3% slower than one leaf per block
    (the clobbers spill the surrounding code).
  - Two leaves interleaved in one asm block: -0.6%, `forward_product` -3.6%; not worth a second
    kernel.
  - Instruction orders: M A alternating, E or O first, `vshufps` + `vpshufd` for the final
    interleave: within 1% of the kept order. Intrinsics with the same bottoms: 2-5% slower.
- Counted, not built: Karatsuba (the wrap needs w-scaled and plain halves of the same
  sub-products: 64 products again); leaves mod x^4 - w (an in-register level costs ~14 ops per
  vector in every transform, the product saves ~8 per vector); FMA on doubles (30-bit operands
  need two limbs: 32 FMAs against 30 integer ops); one Montgomery step for K products in
  `inverse_product_sum` (16 P^2 + 2^32 P > 2^64); q by one `vpmulld` (-1 multiply, +2 shifts,
  +1 blend; estimated neutral).
- Whole process (`judge.py bench`, 21 rounds, ratios new/main): `lc-amd` inv 0.9550, exp 0.9407,
  log 0.9442, pow 0.9311, sqrt 0.9490 (exp 18.65 -> 17.53 ms, pow 29.33 -> 27.33 ms); `lc-intel`
  inv 0.9800, exp 0.9841, log 0.9700, pow 0.9768, sqrt 0.9826. All official tests pass
  (`judge.py test`, `lc-amd`).
- Merged as #158 (CI: exp 0.9384, inv 0.9493, log 0.9529, pow 0.9402, sqrt 0.9561; all 5
  0.9473). Judged: exp [409327](https://judge.yosupo.jp/submission/409327) AC 18 ms (one case
  at 18, the rest <= 17); pow [409328](https://judge.yosupo.jp/submission/409328) AC 32 ms
  with two launch spikes, large cases 25-27 ms (clean 27, was 29).

2026-10-09, claude (issue #69, inv_of_formal_power_series_sparse):
- New `sparse.hpp` (above) and its tests: the recurrence against its definition for 3000 random
  tap sets (short, long, mixed, block edges, 16 taps at P - 1), next() in random lengths into an
  array and into a ring; each width 1..15 over 20000 coefficients; the reduction at its bounds
  (12 (P - 1)^2 unfolded, 2^64 - 1 folded). With the unfolded bound raised to 14 the tests fail;
  random recurrences alone did not catch it.
- Kernel steps, solve phase of small_dense_02 (w = 7, 10^6 coefficients) in process, ms: sums in
  a struct by reference and a call per block (GCC kept them on the stack) 2.16; sums in
  registers, 32 per step 0.85; 64 per step, unrolled per width, an empty asm after each column
  (GCC otherwise formed all products first and spilled 14 per block) 0.68; the block holding
  the next state first (it was ~285 instructions into a step, past the 256-entry reorder
  buffer) 0.53. In memory, w = 7: 0.711 -> 0.557 -> 0.463 ns per coefficient.
- Not kept: the state in registers, lanes broadcast by `vpermd` (0.711 at 32 per step, 0.766 at
  64; GCC spilled the state); broadcasts from memory cost no vector uop and their store
  forwarding is off the critical path once the state block comes first. Each column as asm in
  the order M M A A (the leaf product's finding above): w = 7 0.480 against 0.466 for GCC's
  M A M A, two alternated runs (the multiplies here take their column from memory).

2026-10-09, claude (issue #67, composition_of_formal_power_series):
- New `composition.hpp` (Composition above); no existing header changed. Tests: compose against
  Horner's rule for n = 1 .. 160 (four kinds of g, among them leading zero runs), f and g shorter
  and longer than n, g = 0 and g = x; up to 2^17 + 1 by identities at random coefficients
  (f = sum c^i y^i: h (1 - c g) = 1; the chain rule h' = (f' o g) g'; f = y^2); scratch filled
  with garbage first. Nine mutations (wraps, signs, truncation, CRT factor, each special level)
  fail them.
- Steps and measurements: problems/polynomial/composition_of_formal_power_series/notes.md.
  Whole process at N = 8000: 3.44 ms (first version) -> 2.79 ms, floor 1.25 ms. Merged as #164;
  judged [409332](https://judge.yosupo.jp/submission/409332): AC 11 ms with a launch spike, clean
  3 ms (record 9 ms).
- Leaf-product findings: windows filled and read at once stall on store forwarding (86.5 against
  47.1 us for the same products with windows stored long before); GCC kept a 4-term loop rolled
  with its operands on the stack until `#pragma GCC unroll`; 8 pairs per step in transposed
  form beat one pair per vector (50.1 against 58.6 us per inverse of 2^15).
- For the owner lane: read access to Transform's twiddle tables would save Tables' copies (2 x
  2^lg / 8 words), and `times` could take per-lane factors.

2026-10-09, claude (issue #62, round 2):
- `cyclic_product(..., Half, c)`: output times c through the scale. The inverse's step negates
  in the scale and copies the half into g (was: a scalar negation loop, 0.135 ms at 2^19);
  `inverse_step` split out. exp: the negation of h[m/2, m) folded the same way, then a copy
  (was: a vector negation pass).
- Tried, not kept: the top level's inverse writing its half straight into a destination (a
  `HalfOut` sink with a masked tail store) instead of in place and a copy. In process the same
  or faster (inverse of 2^18, `lc-intel`: direct 4.548 ms, in place + copy 4.558, main's
  negation 4.625). Whole process on `lc-intel` (`judge.py bench`, 31 rounds): sqrt with its five
  copies replaced 1.0146 against the copies' 0.9940; pow 1.0151 with direct writes in the inverse
  and exp, 1.0102 with them in the inverse only, 0.9995 with neither. `lc-amd`: neutral (sqrt
  1.0020). Cause not found; guess: write-allocate traffic of the scattered half stores into a
  cold array, which `memmove`'s streaming copy avoids. Destinations are kept for in-place halves
  only.
- `judge.py bench`, 21 rounds, ratios new/main, `lc-amd` (`lc-intel`): inv 0.9618 (0.9748), exp
  1.0013 (0.9969), log 0.9954 (0.9997), pow 0.9985 (1.0015), sqrt 1.0012 (0.9946), composition
  1.0018 (0.9996). All official tests pass on `lc-amd`; `test.cpp` passes at -O2 and
  ASan/UBSan, `-march=native` and `-march=x86-64-v3` (`lc-intel`).
- Kernel costs on `lc-amd`, cycles per vector (3.48 GHz), 256-vector tile in L1 (scratch probe):
  lib/ntt `forward` kernel at h = 64, 16, 4: 3.61, 3.71, 4.27 (call overhead at h = 4); `inverse`
  3.63, 3.67, 3.85; `ForwardBottom` 5.98 (GCC: 71 vector ops and ~26 scalar ops for the twiddle
  slots per group); `InverseBottom` 4.39; `ProductBottom` 30.0 (326 vector ops per group, 128 on
  the multiply pipes: bound ~20.4); `InverseProductBottom` 25.4. The same radix-4 butterfly in
  intrinsics: 4.31 (forward), 4.11 (inverse) against the asm's 3.64, 3.62. At 2^19 in place:
  forward top level (`Source`) 4.17, levels h = 4096, 1024, 256: 3.49, 3.65, 3.77; tiles 18.3;
  the whole forward 34.3, `cyclic_product` 86.8.

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
- Linear recurrences in blocks by jump matrices (the state times A = M^t rows) and the impulse
  response of the short part: standard linear algebra, derived here; no code read.
- Montgomery reduction: P. Montgomery, "Modular multiplication without trial division",
  Math. Comp. 44 (1985).
- Composition: K. Kinoshita, B. Li, "Power Series Composition in Near-Linear Time", FOCS 2024,
  https://arxiv.org/abs/2404.05177 (power projection by Graeffe steps in x with y as the
  coefficient ring, composition as its transpose). Layout, leaf-level Graeffe and transposed
  steps derived and written here; no code read.
