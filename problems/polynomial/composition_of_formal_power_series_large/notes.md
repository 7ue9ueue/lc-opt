# composition_of_formal_power_series_large

N <= 131072 coefficients of f and of g (g[0] = 0) mod 998244353; print f(g) mod x^N. 10 s.
Slowest tests: max_random, hack, hack2 and random_00, _01 (N = 131072; hack and hack2 start g
after a run of zeros, which the algorithm does not use). mid: N <= 8000; small: N <= 10.

Best judged: ours, 39 ms, no spike: [409504](https://judge.yosupo.jp/submission/409504)
(`main.cpp` of #238). Clean score 35 ms: [409517](https://judge.yosupo.jp/submission/409517)
(#242, judged 43 ms from two +9 ms launch spikes).
Record when opened (issue #86): 72 ms.

## Design

- Kinoshita and Li's composition, the transpose of power projection: `lib/poly/composition.hpp`
  (design and costs in lib/poly/notes.md, Composition), as composition_of_formal_power_series.
  m = 2^17 for N = 131072: 17 levels; levels 2 .. 13 bivariate with transforms of length
  4m = 2^19 (2 MiB), levels 0, 1, 14, 15, 16 one-dimensional. The forward pass stores each
  level's transforms: 12 x 2 MiB and less for the others.
- Input by `lib/io/bulk32.hpp`; output in fixed-width fields
  (`problems/convolution/convolution_mod/fields.hpp`, judge-specific: the checker compares tokens),
  its text in f's span once compose is done. One `poly::Arena` (huge pages) for everything. The
  program runs from `.preinit_array` and ends with `_exit`.

## Floor

`lc-amd`, whole process (judge's runner and flags, 15 interleaved rounds, score = slowest of
max_random_00, max_random_03, hack2_00, hack_02, random_00; scratch `rawbench.py`): read and
write only (main.cpp with h = f) 2.89 ms; main.cpp 38.7 ms. `tools/floor.c` (no parsing):
1.6 ms. With the bulk input and the fields output (21 rounds): 2.37 ms against 2.91.

In process (N = 131072, warm, median of 10), round 1's start: compose 34.0 ms. Per generic
level (s = 1 .. 14, µs): forward of Q_s at 4m 480-630 plus the Graeffe bottom ~350; inverse of
V at 2m ~250; forward of P at 2m ~260; inverse of R at 4m 460-580 plus the product bottom ~370.
Sums: forward pass 16.0 ms (level 0 0.73, generic 15.0, last levels 1.1), backward 17.1 ms
(last levels 1.0, generic 15.4, level 0 0.7). The bottoms were ~10 ms of 34.

After #242 (`main.cpp` 34.75 ms whole process): compose 29.97 ms warm, 31.7 ms on the
first call (page faults: ~100 µs per 2 MiB huge page on `lc-amd`, ~20 pages). Split (ms):
generic levels 2 .. 13 forward 9.09 (with the Graeffe bottom ~3.0) + 2.76, backward 2.89 + 9.53
(the product bottom ~3.6); levels 0: 0.60 + 0.68; 1: 0.73 + 0.81; 14: 0.51 + 0.75; 15, 16: 0.61
+ 0.94.

## Log

- 2026-10-10, claude (round 1): first solution: composition_of_formal_power_series's
  solution.cpp unchanged (lib/poly/composition.hpp handles any N). 32/32 official tests
  (`judge.py test`, `lc-amd`, slowest 56.7 ms on a first run, others 38-41 ms); `stress.py` 400
  rounds. Merged as #238. Submitted: [409504](https://judge.yosupo.jp/submission/409504) AC
  39 ms, 43.9 MiB, clean 39 ms (`tools/spikes.py`: no spike; 13 cases within 9 ms of the max).
- 2026-10-10, claude (round 1, lib/poly owner lane): lib/poly/composition.hpp, in process at
  N = 131072 (`lc-amd`, medians; each step kept):
  - Bottoms in power projection's form (`graeffe_leaf`, one sum and one reduction per output
    coefficient, even lanes then odd lanes; per 8 pairs 684 instead of 935 instructions for the
    Graeffe, 893 instead of 1070 for the product): compose 34.0 -> 31.6.
  - Level 1 one-dimensional (`first_level`, `first_level_transposed`): 31.6 -> 31.2 (level 1
    2.44 -> 1.59 ms). Needs m >= 256 (the product bottom's tiles).
  - Level T - 3 one-dimensional in y (`third_last_level`, 6 products at m/4 forward, 16 backward):
    31.2 -> 30.0 (forward 2.04 -> 1.13 ms, backward 2.13 -> 1.70 ms).
  - `next_level` and level 1 no longer zero the x >= L/2 halves (the pruned forward does not read
    them): next_level 235 -> 6 µs; compose within noise.
  - Whole process (`judge.py bench`, 21 rounds, against #238): `lc-amd` 0.8973 (38.70 -> 34.75 ms),
    `lc-intel` 0.9219. PR #242.
  - Checks: 32/32 official tests (`lc-amd`); `stress.py` 400 rounds (`lc-intel`); lib/poly
    `test.cpp` at -O2 (x86-64-v3 and native) and ASan/UBSan.
- 2026-10-10, claude (round 1): submitted #242's `main.cpp`:
  [409517](https://judge.yosupo.jp/submission/409517) AC 43 ms, 44.0 MiB. hack_02 and
  max_random_03 took 43 ms, the 19 other large cases 33-35: two +9 ms launch spikes, clean 35 ms
  (`tools/spikes.py` compared it with 409504's 38 ms and flagged neither).
- 2026-10-10, claude (round 1, second lib/poly change): composition's temporaries live in level
  slots not filled yet (forward: V of level s in level s + 1's slot, inverted in place; level 0's
  v and level 1's products in level 3's; level T - 3's in level T - 1's) or no longer read
  (backward: each level writes R into a dead slot), so the two work spans of 4m are gone.
  AnonHugePages at the end of compose 42 -> 38 MiB (a huge page's first touch costs ~100 µs on
  `lc-amd`); in process 29.98 -> 29.85 ms warm, first call 31.7 -> 31.5. Bulk input and fields
  output: whole process 0.991-0.995 against #242 (`rawbench.py`, 21 rounds; the text after h or
  in f's span, equal within noise). Both together (`judge.py bench`, 21 rounds): `lc-amd` 0.9887
  (34.85 -> 34.43 ms), `lc-intel` 0.9846.
- Merged as #248 (CI 0.9876 for this problem).
- 2026-10-10, claude (round 1, third lib/poly change): composition's coefficient loops in AVX2
  (GCC left them scalar at -O2; the final h loop took 125 µs alone). compose in process 29.86 ->
  29.40 ms. `judge.py bench` (21 rounds, against #248): `lc-amd` 0.9879 (34.41 -> 34.03 ms),
  `lc-intel` 0.9912.
- Tried, not kept: level T - 4 one-dimensional too (column levels for 16 columns: 28 products
  forward, 64 backward, at m/8). In process at m = 2^17: forward 766 µs, backward 1249 µs against
  ~1030 each for the generic level; compose 29.8 ms either way. Its leaf products cost 15-18 µs
  per 2048 leaves in sums of 8 (~31 cycles a leaf: 16 operand arrays of 64 KB exceed L2) against
  13 for a single product. Preparing the windows 2 or 3 groups ahead instead of 1 changed
  nothing (one product: 25.3, 24.9, 25.0 µs including the inverse of 12.3).
- Next: the y levels by doubling (from V's transform at 2Y points, Q_(s+1) at the other 2Y
  points by an inverse and a forward in y of the truncated rows; the first half of the next
  level's transform then needs no y levels): counted ~9-11% of the transform work, ~1.5-2 ms;
  radix-4 subtrees do not align with the x blocks at odd s. Level T - 4 one-dimensional: ~0.5 ms
  (estimate). Pack the small levels in memory (~3 huge pages, ~0.3 ms) and run the backward
  pass in dead levels' storage (2 work spans of 2 MiB).

## Sources

- K. Kinoshita, B. Li, "Power Series Composition in Near-Linear Time", FOCS 2024,
  https://arxiv.org/abs/2404.05177 (the algorithm). Derived and written here; no code read.
- Brute force: Horner's rule, written here.
