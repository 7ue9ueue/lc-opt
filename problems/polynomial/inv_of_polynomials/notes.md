# inv_of_polynomials

f (N coefficients) and g (M coefficients), N, M <= 50000, mod 998244353: print h = 1 / f mod g
with deg h < deg g, or -1 if gcd(f, g) != 1. 10 s. Tests: max_random (N = M = 50000, gcd 1),
random (N, M up to 50000 at random: f mod g, then a long first quotient when N < M), abnormal
(degree ~1000, remainder sequences with quotients of degree up to 20), examples (M = 1: h = 0;
a common factor: -1). The checker compares tokens.

Best judged: none yet. Record when opened (issue #82): 104 ms.

## Design

- `poly::inverse_mod` (lib/poly/gcd.hpp, design in lib/poly/notes.md, Gcd): b = f mod g
  (`poly::divide`), then the jump of deg g on (g, b) by half-gcd recursion; h = R[0][1] / c for
  R (g, b) = (c, 0), c the constant gcd (coefficient 0 of each entry is tracked, so only R[0][1]
  is computed at the top and only row 0 along the right spine).
- I/O: `io::read_bulk`; output in 10-byte fixed-width fields
  (problems/convolution/convolution_mod/fields.hpp, judge-specific). `RUN_EARLY` (lib/run).

## Floor

`tools/speed.py bench` on `lc-bench` (floor: map the input, write an output of the expected
size), 7 rounds, commit c08ddec: max_random 1.02-1.04 ms, the others 0.7-0.9 ms. The score is set
by the five max_random tests (all within 0.1 ms of each other).

## Log

- 2026-10-10, claude (round 1): first solution and lib/poly/gcd.hpp.
  - Checks: 18/18 official tests (`judge.py test`, `lc-amd`); `stress.py` 400 rounds against
    `brute.cpp` (`lc-intel`, -march=x86-64-v3; N, M <= 2000, common factors, g | f, remainder
    sequences with quotients up to degree 30, sizes 1 and 2); lib/poly tests at -O2 and
    ASan/UBSan (`lc-amd`): jumps against the extended Euclidean algorithm with every direct
    threshold, inverses up to 500 by brute force, five large pairs by (f h - 1) mod g = 0.
  - v1 (balanced split k1 = ceil(k / 2), base Euclid at k <= 32 in scalar code, every transform
    recomputed): max_random 27 ms (`judge.py test`, `lc-amd`); random_04 425 ms (its first
    quotient, degree 18713, by long division: now `divide` above 2^12 terms).
  - In process on max_random_00 (`lc-amd`, inverse_mod with a fresh arena, min of 20): v1 25.2
    ms; transform reuse (children keep their product transforms, parents double them with
    `forward_upper`), only R[0][1] at the top and row 0 on the spine, vector Euclid: 19.4;
    vector folds and unwraps in apply: 18.1; Euclid up to k = 64: 17.2; row products sharing
    windows, pipelined: 16.5; Montgomery three-term Euclid steps: 16.2; first jumps at powers of
    two: 13.9 (random_03 7.5 -> 6.6; random_04 4.35 -> 5.0: its long first quotient now sits in
    a full 4-entry combine of length 32768).
  - Few wrapped coefficients in the apply (the top node, n = 49999, k = 32768: 848) by a short
    product of length bit_ceil(2 low - 1) instead of length L = 32768: 14.24-14.36 -> 14.06-14.08
    ms (`lc-bench`, in process, medians of 3 alternating runs).
  - Tried, no gain: the apply's products as separate passes sharing the windows of a, b, then
    inverses (16.6 against 16.5 ms for the fused `inverse_product_sum`); kDirect 96, 97, 98, 128
    (within 1% at n = 50000; 128 is 4% slower at 65535).
  - `speed.py bench` (`lc-bench`, 7 rounds, c08ddec): max_random 16.04-16.10 ms (v1 with
    kDirect 64, d024926: 18.5); random_03 8.28, random_04 6.70, abnormal 1.5-1.8.
  - Profile (`lc-intel`, before the power-of-two split): leaf products 43%, Euclid base 16%,
    transforms ~30%. Per node of 2k coefficients: ~16 leaf products and ~15 transforms of length
    k; each level costs 16-28 ns per degree.
  - Next: pointwise products (transforms to single points; leaf products are half the time);
    smaller scratch (12-13 MiB RSS against ~6 for small cases); a faster base (two steps per
    pass); the right spine's combines (only row 0, 4 products at twice the length).
