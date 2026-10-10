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

## Log
