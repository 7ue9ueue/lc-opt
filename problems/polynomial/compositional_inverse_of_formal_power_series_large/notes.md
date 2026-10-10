# compositional_inverse_of_formal_power_series_large

N <= 131072 coefficients of f (f[0] = 0, f[1] != 0) mod 998244353; print g with f(g) = x mod x^N.
10 s. Slowest tests: max_random (5), max_identity, random_00 (N = 131072 or near). mid: N <= 8000;
small_degree: N <= 11.

Best judged: ours, 33 ms, no spike: [409549](https://judge.yosupo.jp/submission/409549) (#266's `main.cpp`).
Record when opened (issue #87): 93 ms.

## Design

- As compositional_inverse_of_formal_power_series (#68): Lagrange inversion, (N - 1) [x^(N-1)] f^k
  = k [x^(N-1-k)] (x / g)^(N-1). Power projection gives a[k] = [x^(N-1)] f^k; H = (x / g)^(N-1) /
  (N - 1) by one division by index; g / x = (H / H[0])^(-1 / (N-1)) / f[1] by `poly::power`.
  `lib/poly/compositional_inverse.hpp`, `lib/poly/projection.hpp`; design in lib/poly/notes.md.
- m = 2^17 for N = 131072: 17 levels; levels 0 and 1 one-dimensional in x, 2 .. 13 bivariate
  (transforms of length 4m = 2^19), 14, 15 and 16 one-dimensional in y. Forward only: two arrays
  of 4m and a work span of ~2m.
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
  changes one output coefficient for N > 1000 fails it. Merged as #258.
- Submitted #258's `main.cpp`: [409541](https://judge.yosupo.jp/submission/409541) AC 47 ms,
  9.9 MiB: max_random_01 47 and max_random_03 46 ms, the other large cases 36-37: two launch
  spikes, clean 37 ms (`tools/spikes.py` had no earlier run to compare with).
- lib/poly/projection.hpp (this round, lib/poly composition and owner lanes): level T - 3
  one-dimensional in y (as composition's), level 1's coefficient loops in AVX2, no zero fills of
  the row halves the pruned forwards do not read. In process (`phases.cpp`): 34.5 -> 32.3 ms;
  levels 0 and 1 2.01 -> 1.75, levels T - 3 .. T - 1 ~4.2 -> 2.7 (columns 0.06, 14 forwards at
  m/4 0.38, Q' products 0.26, P' products 0.59; then 6 forwards at m/2 0.37, products at m/2
  0.45, the product at m 0.58). `judge.py bench` (21 rounds, against #258): `lc-amd` 0.9466
  (36.89 -> 34.92 ms), `lc-intel` 0.9607. 28/28 official tests; `stress.py` 400 rounds. Merged as
  #260 (CI 0.9557, 0.9507, 0.9633).
- lib/poly/projection.hpp, doubling between generic levels (lib/poly/notes.md, Power projection):
  each level's transforms of length 4m from the previous level's leaves, the first half without
  y levels. In process (alternating runs): 32.35 -> 30.50 ms (generic levels 23.17 -> 21.27).
  `judge.py bench` (21 rounds, against the previous change): `lc-amd` 0.9493 (35.28 -> 33.49 ms),
  `lc-intel` 0.9165. 28/28 official tests; `stress.py` 400 rounds. Merged as #266 (CI 0.9512,
  0.9482, 0.9490).
- Submitted #266's `main.cpp`: [409548](https://judge.yosupo.jp/submission/409548) AC 42 ms:
  max_random_02 42 ms against 31-33 for the other large cases, a launch spike (`tools/spikes.py`
  compared it with 409541's slower run and did not flag it), clean 33 ms; the same file
  [409549](https://judge.yosupo.jp/submission/409549) AC 33 ms, 10.0 MiB, no spike. New best
  judged: 33 ms (was 47, 409541). 3 of the session's 5 submissions used.
- Where the time goes now (`lc-amd`, in process 30.5 ms at N = 131072): levels 0, 1 1.78 ms;
  level 2's forwards 1.61; doubling 19.2 (P 6.66, Q with ProjectionBottom 12.58); level 13's
  inverses 0.42; levels 14 .. 16 2.71; division 0.16; power 4.56 (log_derivative 1.92, exp_newton
  2.61). Profile (`lc-intel`, `perf`, x86-64-v3, 30 runs of max_random_00): forward kernels ~36%
  (radix-4 levels 14%, bottoms 9%, h = 4 8%, columns 4%), inverse kernels ~18%,
  ProjectionBottom 16.5%, the product bottoms of the last levels and power ~7%, page zeroing 1%.
- Next (counted, not built): level T - 2's transforms of length m/2, first halves from the
  product sums in the transform domain (~0.15 ms); level T - 3's column transforms, first halves
  from level T - 4's leaves by inverse x levels and 8-point inverses (~0.2 ms); `power` (4.5 ms)
  is exp/log (lib/poly owner lane).

## Sources

- K. Kinoshita, B. Li, "Power Series Composition in Near-Linear Time", FOCS 2024,
  https://arxiv.org/abs/2404.05177 (power projection; compositional inverse by Lagrange inversion
  and one power). Derived and written here; no code read.
- Lagrange inversion: n [x^n] f^k = k [x^(n-k)] (x / g)^n (standard; R. Stanley, Enumerative
  Combinatorics vol. 2, 5.4).
- Brute force: powers of f and the triangular system of g(f) = x, written here. Closed forms in
  `stress.py`: Catalan numbers (x - x^2), e^x - 1 (log(1 + x)), Lambert's k^(k-1) / k! (x e^-x).
