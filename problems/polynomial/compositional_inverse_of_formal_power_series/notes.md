# compositional_inverse_of_formal_power_series

N <= 8000 coefficients of f (f[0] = 0, f[1] != 0) mod 998244353; print g with f(g) = x mod x^N.
10 s. Slowest tests: max_random, max_identity and most random (N = 8000 or near). Small tests
N <= 11.

Best judged: none yet.
Record when opened (issue #68): 14 ms.

## Design

- Lagrange inversion: (N - 1) [x^(N-1)] f^k = k [x^(N-1-k)] (x / g)^(N-1). Power projection gives
  a[k] = [x^(N-1)] f^k for all k; then H = (x / g)^(N-1) / (N - 1) mod x^(N-1) by one division by
  index, and g / x = (H / H[0])^(-1 / (N-1)) / f[1] by `poly::power` (log, then exp).
  `lib/poly/compositional_inverse.hpp`.
- Power projection: Kinoshita and Li, forward direction (`lib/poly/projection.hpp`, design in
  lib/poly/notes.md, Power projection): composition.hpp's levels Q_s, with the numerator P_s
  carried along, so no level is stored. m = 8192 for N = 8000: level 0 one-dimensional in x,
  levels 1 .. 10 bivariate (Kronecker layout, transforms of length 4m), levels 11 and 12
  one-dimensional in y.
- `lib/io` input and output, one `poly::Arena`, the program runs from `.preinit_array` (as
  composition_of_formal_power_series).

## Floor

`lc-amd`, whole process (judge's runner and flags; 15 interleaved rounds, score = slowest of
max_random_00, max_random_03, random_04, max_identity_00): read and write only (main.cpp with
g = f) 1.22 ms. main.cpp 2.77 ms.

## Log

- 2026-10-10, claude (round 1): first solution, with new `lib/poly/projection.hpp` and
  `lib/poly/compositional_inverse.hpp` (issue #95, composition lane). In-process times at
  N = 8000 on `lc-amd` (medians of 41), each step kept unless noted:
  - v1: general weight w (P_0 = sum w[i] x^(m-1-i), level 0 by a transform of P_0); bottom of Q's
    transform at 4m computing V = Q(x) Q(-x) (Graeffe, as LevelBottom) and W = odd part of
    P(x) Q(-x) (32 products per leaf, wrapped terms reduced, then times s), CRT as LevelBottom.
    1759 us: level 0 78, P forward 293, Q forward with the bottom 741, inverses 263, layouts 17,
    last levels 99, division by index 10, power 251. Whole process 3.00 ms.
  - Weight x^(n-1) only (the only one Lagrange needs): P_1 is a shift of g's coefficients, level
    0 needs only the Graeffe of g: 78 -> 42 us.
  - Bottom: wrapped terms take t c_j (6 Shoup products per leaf set) instead of a reduction and a
    product per output; CRT as (A + B) + u^4 (A - B) / s with s^-1 from the inverse table, the
    factor 2^-31 undone by the inverses' scale: Q forward 741 -> 670. Total 1643.
  - Not kept: the leaf functions inlined into the pairs (545 stack moves): 670 -> 736.
  - Pruned transforms (see lib/poly/notes.md): the y levels skip the zero x-padding columns
    forward and the discarded x >= L/2 columns inverse. In-process A/B (two library copies in one
    binary, interleaved): P forward 294 -> 273, Q 729 -> 720, inverses 261 -> 224. Per level
    (forward of length 4m, plain bottom): -12% for s <= 6, none or slower for s >= 9 (31.1 against
    27.5 us at s = 10: columns of one vector); one radix-4 level by intrinsics 5.18 us against the
    asm kernel's 4.34.
  - Leaf sums per lane parity (64-bit products of the even lanes as given, the odd lanes after a
    shift) instead of `Lanes`/`Wide`: in one function 810 against 720 (worse); in one function
    per leaf kind 671 against 717.
  - GCC kept `transpose` (composition.hpp) out of line, with vzeroupper and every vector spilled
    around the 10 calls per 8 pairs: `[[gnu::flatten]]` on the pairs and leaves functions: Q
    forward 671 -> 625 (measured with `transpose` always inlined; flatten 0.993 against that).
    composition.hpp left unchanged: always inlined, compose was neutral (0.998 in process, 0.993
    `judge.py bench`), and a neutral change fails CI's gate half the time.
  - Unpruned below 64 vectors forward (the bottoms' groups; was 16): 0.991. Inverse at 64: no
    change, kept 16.
  - Rebased onto #170 (generated kernels for h = 4 and h = 1): whole process 2.77 ms against 3.03
    for v1 in the same run (floor 1.22).
  - power at N - 1 = 7999: 251 us (log_derivative 105, exp 146; the inverse alone 84).
  - Checks: 23/23 official tests (`judge.py test`, `lc-amd`, slowest 2.8 ms); `stress.py` 1000
    rounds against `brute.cpp` (N <= 400, g(f) = x solved coefficient by coefficient); lib/poly
    tests at -O2 (native on `lc-intel`, x86-64-v3) and ASan/UBSan. A mutation (sign of W at level
    0) fails 1116 checks.

## Sources

- K. Kinoshita, B. Li, "Power Series Composition in Near-Linear Time", FOCS 2024,
  https://arxiv.org/abs/2404.05177 (power projection by Graeffe steps; compositional inverse by
  Lagrange inversion and one power). Derived and written here; no code read.
- Lagrange inversion: n [x^n] f^k = k [x^(n-k)] (x / g)^n for g the compositional inverse of f
  (standard; e.g. R. Stanley, Enumerative Combinatorics vol. 2, 5.4).
- Brute force: powers of f and the triangular system of g(f) = x, written here.
