# composition_of_formal_power_series_large

N <= 131072 coefficients of f and of g (g[0] = 0) mod 998244353; print f(g) mod x^N. 10 s.
Slowest tests: max_random, hack, hack2 and random_00, _01 (N = 131072; hack and hack2 start g
after a run of zeros, which the algorithm does not use). mid: N <= 8000; small: N <= 10.

Best judged: none yet.
Record when opened (issue #86): 72 ms.

## Design

- Kinoshita and Li's composition, the transpose of power projection: `lib/poly/composition.hpp`
  (design and costs in lib/poly/notes.md, Composition), as composition_of_formal_power_series.
  m = 2^17 for N = 131072: 17 levels; levels 1 .. 14 bivariate with transforms of length
  4m = 2^19 (2 MiB). The forward pass stores each level's transform: 14 x 2 MiB.
- `lib/io` input and output, one `poly::Arena` (huge pages) for everything. The program runs from
  `.preinit_array` and ends with `_exit`.

## Floor

`lc-amd`, whole process (judge's runner and flags, 15 interleaved rounds, score = slowest of
max_random_00, max_random_03, hack2_00, hack_02, random_00; scratch `rawbench.py`): read and
write only (main.cpp with h = f) 2.89 ms; main.cpp 38.7 ms. `tools/floor.c` (no parsing):
1.6 ms.

In process (N = 131072, warm, median of 10): compose 34.0 ms. Per generic level (s = 1 .. 14,
µs): forward of Q_s at 4m 480-630 plus the Graeffe bottom ~350; inverse of V at 2m ~250;
forward of P at 2m ~260; inverse of R at 4m 460-580 plus the product bottom ~370. Sums: forward
pass 16.0 ms (level 0 0.73, generic 15.0, last levels 1.1), backward 17.1 ms (last levels 1.0,
generic 15.4, level 0 0.7). The bottoms are ~10 ms of 34.

## Log

- 2026-10-10, claude (round 1): first solution: composition_of_formal_power_series's
  solution.cpp unchanged (lib/poly/composition.hpp handles any N). 32/32 official tests
  (`judge.py test`, `lc-amd`, slowest 56.7 ms on a first run, others 38-41 ms).

## Sources

- K. Kinoshita, B. Li, "Power Series Composition in Near-Linear Time", FOCS 2024,
  https://arxiv.org/abs/2404.05177 (the algorithm). Derived and written here; no code read.
- Brute force: Horner's rule, written here.
