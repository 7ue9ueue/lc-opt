# composition_of_formal_power_series

N <= 8000 coefficients of f and of g (g[0] = 0) mod 998244353; print f(g) mod x^N. 10 s.
Slowest tests: max_random, hack, hack2 and most random (N = 8000 or near; g with a run of
leading zeros in hack). Small tests N <= 10.

Best judged: ours, 11 ms: [409332](https://judge.yosupo.jp/submission/409332) (`main.cpp` of #164),
with a +9 ms launch spike on random_02 (11 ms, its peers 2; `tools/spikes.py`): clean score
3 ms.
Record when opened (issue #67): 9 ms.

## Design

- Kinoshita and Li's composition, the transpose of power projection: `lib/poly/composition.hpp`
  (design and costs in lib/poly/notes.md, Composition). m = 8192 for N = 8000: 13 levels, each a
  Graeffe step on the bivariate Q_s(x, y) (forward pass, stored as transforms) and a transposed
  step on P (backward pass), with transforms of length 4m and 2m on Kronecker layouts. Levels 0,
  T - 2 and T - 1 run as one-dimensional products.
- `lib/io` input and output (8000 values: plain `write_array`). One `poly::Arena` for f, g, h,
  the tables and the scratch. The program runs from `.preinit_array` and ends with `_exit`.

## Floor

`lc-amd`, whole process (judge's runner and flags, `rawbench.py`-style: 15 interleaved rounds,
score = slowest of max_random_00, max_random_03, hack_02, random_04): read and write only
(main.cpp with h = f instead of the composition) 1.25 ms; main.cpp 2.79 ms. In process
(N = 8000, warm, median of 41): compose 1.46 ms (forward pass 0.71, backward 0.75).

## Log

- 2026-10-09, claude (round 1): first solution, with `lib/poly/composition.hpp` (issue #95).
  In-process times at N = 8000 on `lc-amd` (medians of 41 calls), each step kept unless noted:
  - v1: x = z, y = z^(2L) Kronecker layout. Forward: transform of Q_s at 4m, leaf pairs split into
    E, -O, u O (Q = E(x^2) + x O(x^2)), V = E^2 - u O^2 by `inverse_product_sum`. Backward: two
    middle products (E and -O) at 2m by `inverse_product`, outputs interleaved. 2104 us (forward
    1149, backward 955); whole process 3.44 ms.
  - Graeffe in the forward bottom: per leaf pair a(z) a(-z) mod (z^8 - s) as e^2 - u o^2 mod
    (u^4 - s), 8 pairs per step in transposed form (20 products per leaf instead of 64), then the
    CRT to V mod (u^8 - s^2). Forward pass 1149 -> 1022.
  - Backward as one product R = P(z^2) Q_s(-z) at 4m: the reversals of the middle product cancel,
    R's upper half is the next level's input (no interleave), and P(z^2) mod (z^8 -+ s) has 4
    terms (leaf products of 4 by 8). Leaf product variants (inverse at 4m with the bottom; plain
    inverse 26.9 us):
    - windows [w c, c] filled per leaf and read at once: 86.5 us (store forwarding);
    - shifts by `vpermq` in registers: 70.8 us;
    - windows stored by the forward pass (2x level memory): 55.0 us; with `#pragma GCC unroll`
      on the 4-term loop (GCC had kept it rolled, operands spilled): 47.1 us, compose 1814;
    - a leaf pair per vector (`vpalignr`, `vpshufd`), no windows: 58.6 us forced inline, compose
      1839 (level memory 1x);
    - 8 pairs per step in transposed form (5 transposes per 8 pairs, w folded into the wrapped
      operands): 50.1 us, compose 1698. Kept.
    `ntt::detail::multiply` (`times`) uses one factor for all lanes; per-lane factors need the
    odd lanes' quotients (`times_lanes`).
  - Level 0 one-dimensional (Graeffe of g at 2m; h = p1(x^2) - p0(x^2) g(-x)) and level T - 1
    (one cyclic product of length m): 1555 us. Level T - 2 (four products of length m/2): 1470 us.
  - Forward sources end after row Y (rows Y + 1 .. 2Y - 1 not read or zeroed): 1460 us.
  - Per generic level (1 .. T - 3) now: forward 48 us (transform of 4m with the Graeffe bottom;
    plain 27) + 13 (inverse of 2m); backward 13 (transform of 2m) + 50 (inverse of 4m with the
    product bottom).
  - Whole process (15 rounds, see Floor): v1 3.44 ms, final 2.79 ms, floor 1.25 ms.
  - Checks: 27/27 official tests (`judge.py test`, `lc-amd`, slowest 2.9 ms); `stress.py` 3000
    rounds (N <= 300 against `brute.cpp`, Horner's rule; N up to 8000 by identities); lib/poly
    tests at -O2 (native on `lc-intel`, x86-64-v3) and ASan/UBSan; main.cpp under ASan/UBSan.
    Mutations caught by the tests: 9 of 9 (wraps, signs, truncation, CRT factor, each special
    level).
  - Merged as #164 (CI: correctness only, no baseline). Submitted its `main.cpp`:
    [409332](https://judge.yosupo.jp/submission/409332) AC 11 ms, 4.8 MiB; clean score 3 ms
    (spikes on random_02 11 ms and small_06 9 ms; P(clean run) 0.24, since all 27 cases lie
    within 9 ms of the slowest).
- Next: level 1 one-dimensional (Y = 2, ~50 us estimate); pruned y-levels in the transforms of
  generic levels (x-padding bit below the y-bits: y-levels need only half the columns; estimate
  up to 18% of transform time, needs radix-4 kernels whose stride differs from their count).

## Sources

- K. Kinoshita, B. Li, "Power Series Composition in Near-Linear Time", FOCS 2024,
  https://arxiv.org/abs/2404.05177 (the algorithm: power projection by Graeffe steps in x with y
  as coefficients, and composition as its transpose). Derived and written here; no code read.
- Brute force: Horner's rule, written here.
