# compositional_inverse_of_formal_power_series_large

N <= 131072 coefficients of f (f[0] = 0, f[1] != 0) mod 998244353; print g with f(g) = x mod x^N.
10 s. Slowest tests: max_random (5), max_identity, random_00 (N = 131072 or near). mid: N <= 8000;
small_degree: N <= 11.

Best judged: none yet.
Record when opened (issue #87): 93 ms.

## Design

- As compositional_inverse_of_formal_power_series (#68): Lagrange inversion, (N - 1) [x^(N-1)] f^k
  = k [x^(N-1-k)] (x / g)^(N-1). Power projection gives a[k] = [x^(N-1)] f^k; H = (x / g)^(N-1) /
  (N - 1) by one division by index; g / x = (H / H[0])^(-1 / (N-1)) / f[1] by `poly::power`.
  `lib/poly/compositional_inverse.hpp`, `lib/poly/projection.hpp`; design in lib/poly/notes.md.
- m = 2^17 for N = 131072: 17 levels; levels 0 and 1 one-dimensional in x, 2 .. 14 bivariate
  (transforms of length 4m = 2^19), 15 and 16 one-dimensional in y. Forward only: two arrays of 4m.
- Input by `lib/io/bulk32.hpp`; output in fixed-width fields
  (`problems/convolution/convolution_mod/fields.hpp`, judge-specific: the checker compares tokens),
  its text in f's span after the inverse. One `poly::Arena` (huge pages). `lib/run/early.hpp`.

## Floor

`lc-amd`, whole process (judge's runner and flags; 15 interleaved rounds; score = slowest of
max_random_00, _02, _04, max_identity_00, random_00; scratch `rawbench.py`): read and write only
(main.cpp with g = f) 1.99 ms; first solution 37.03 ms.

In process (N = 131072, `lc-amd`, warm, mean of 10; scratch `phases.cpp`): 34.5 ms. Levels 0 and 1
2.01; generic levels 2 .. 14: forward of P at 4m 6.32, forward of Q at 4m with ProjectionBottom
13.50, inverses of V and W at 2m 5.27 (+0.55 at level 14, unpruned), next layouts 0.22; levels 15
and 16 1.88; division by index 0.16; power at N - 1 4.56.

## Log

- 2026-10-10, claude (round 1): first solution: compositional_inverse_of_formal_power_series's
  solution.cpp with composition_of_formal_power_series_large's I/O (bulk input, fields output) and
  `RUN_EARLY`; lib/poly unchanged. 28/28 official tests (`judge.py test`, `lc-amd`, slowest
  37.0 ms); `stress.py` 300 rounds (`lc-intel`): brute force up to N = 400, and for N up to 131072
  series with known inverses (a x / (1 - b x), x - c x^2, log(1 + x), x e^-x). A mutation that
  changes one output coefficient for N > 1000 fails it.

## Sources

- K. Kinoshita, B. Li, "Power Series Composition in Near-Linear Time", FOCS 2024,
  https://arxiv.org/abs/2404.05177 (power projection; compositional inverse by Lagrange inversion
  and one power). Derived and written here; no code read.
- Lagrange inversion: n [x^n] f^k = k [x^(n-k)] (x / g)^n (standard; R. Stanley, Enumerative
  Combinatorics vol. 2, 5.4).
- Brute force: powers of f and the triangular system of g(f) = x, written here. Closed forms in
  `stress.py`: Catalan numbers (x - x^2), e^x - 1 (log(1 + x)), Lambert's k^(k-1) / k! (x e^-x).
