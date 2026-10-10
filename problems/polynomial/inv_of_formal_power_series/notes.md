# inv_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] != 0; print the first N coefficients of 1/f.
10 s. Largest tests: max_random_* (N = 500000, transforms up to 2^19).

Best judged: ours, 11 ms: [409370](https://judge.yosupo.jp/submission/409370) (`main.cpp` of #170).
Record when opened (issue #62): 25 ms.

## Design

- `lib/poly/inverse.hpp`'s Newton iteration, 5 transforms of length 2k per step k -> 2k
  (lib/poly/notes.md), with the step in `step.hpp` (round 3): between the two products of a step,
  the first one's inverse top level (upper half) and the second one's forward top level are one
  pass; the second product writes its half straight into g instead of in place and a copy.
- `lib/io` input (`io::read_bulk`, the transposed parser on Zen 3); output in 10-byte fixed-width
  fields (`problems/convolution/convolution_mod/fields.hpp`, judge-specific: the checker compares
  tokens).
- Memory: one `poly::Arena` (huge pages). `step::inverse` computes g mod x^k, k = 2^18; the last
  step (`step::inverse_step`) runs in f's buffer of 2^19 words, which ends up holding g[k, N).
  Arrays: tables 0.5 MiB, f 2 MiB, scratch 2 MiB (the inverse's, then T(g mod x^k)), g mod x^k
  1 MiB, text 0.25 MiB: 3 huge pages, against 5 before. The output is written in two pieces
  (a newline after the first; the checker reads tokens).
- The program runs from `.preinit_array` and ends with `_exit` (`lib/run/early.hpp`).

## Floor

`lc-amd`, max_random_00, in-process phases (ms, `CLOCK_MONOTONIC` stamps, scratch probe): parse 0.84,
tables 0.07, inverse 8.26, format 0.32 (to /dev/null); total 9.5. Whole process 13.2 ms, so start,
`write()` of 5 MB and exit take ~3.7. Floor without the inverse: ~5 ms.

Round 3, `lc-bench`, max_random_00, output to tmpfs, medians of 31 runs (us), main -> round 3:
input map 18, tables 83 -> 78, parse 899 -> 822, inverse to 2^18 3521 -> 3471, last step
3718 -> 3714, output 2054 -> 2060 (format ~0.3 ms, the rest `write()` into tmpfs); sum 10.29 ->
10.17 ms. Whole process ~11.7 ms (`judge.py bench`), so start and exit ~1.4 ms. `bench/` (speed.py,
`tools/floor.c`): floor 3.8 ms on `lc-amd`.

## Log

- 2026-10-09, claude (round 1): first solution, with lib/poly built for it (issue #95).
  - Checks: 25/25 official tests (`tools/judge.py test`); `stress.py` 400 rounds (N <= 3000
    against `brute.cpp`, every tenth round N up to 500000 checked by f g = 1 at random
    coefficients); ASan/UBSan on 8 official cases, file and pipe input; lib/poly tests.
  - `lc-amd`, `tools/judge.py bench`, slowest 3 cases (ms, medians): first version 13.55;
    one-half inputs and outputs at the top levels 13.35 (0.988, 15 rounds); top level reading its
    source directly (no copies) 13.32 (0.982 against the first, 31 rounds); plain transforms
    13.15 (0.990 against the previous, 21 rounds).
  - Last Newton step (2^19), ms: forward of g_k 0.64, product with f 1.77, second product 1.69,
    negation and first touch of g 0.15. Steps before it take as long again.
  - Not kept, measured (details in lib/poly/notes.md): b in window form, lib/ntt's asm
    `bottom_last`, other tile sizes, two-leaf interleaving. Not pursued: Harvey's 13/9 M(n)
    reciprocal and Schoenhage's 3k-length step (estimates in lib/poly/notes.md).
  - Submitted the merged `main.cpp` (#107): [409242](https://judge.yosupo.jp/submission/409242)
    AC 13 ms, 14.9 MiB.
- 2026-10-09, claude (lib/poly round, issue #95): faster leaf products (#158; lib/poly/notes.md):
  `cyclic_product` at 2^19 3.38 -> 3.07 ns per coefficient. `judge.py bench` (21 rounds):
  `lc-amd` 13.11 -> 12.52 ms (0.9550), `lc-intel` 0.9800; CI 0.9493. Not submitted (0.6 ms).
- 2026-10-09, claude (round 2, issue #62).
  - Phases on main, in process, `lc-amd` (scratch probe, medians of 41): inverse 7.48 ms warm,
    8.03 first use (+0.55 ms: faults of 3 huge pages). Last step (2^19): forward 0.61, products
    1.58 and 1.56, negation 0.135; all negations 0.28 ms.
  - Changes: the negation folded into the second product's scale (lib/poly
    `cyclic_product(..., c)`, `inverse_step`); the last step in f's buffer, 3 huge pages instead
    of 5 (see Design).
  - `judge.py bench`, 21 rounds, slowest 3 cases: `lc-amd` 12.58 -> 12.13 ms (0.9618), `lc-intel`
    0.9748. An earlier version that wrote the halves straight into g: `lc-amd` 0.9672, `lc-intel`
    0.9927 (slower on `lc-intel` in sqrt and pow; lib/poly/notes.md).
  - Checks: 25/25 official tests (`lc-amd`); `stress.py` 400 rounds; ASan/UBSan on all 25
    official cases, file and pipe input (`lc-intel`); lib/poly tests (in-place last step for
    n = 33 .. 300 and around powers of two).
  - Merged as #166. CI: inv 0.9577 (EPYC 9V45 0.9570, 7763 0.9586, 0.9573); all 6 problems 0.9902.
  - Transform kernels (lib/poly/notes.md): generated asm loops for the levels h = 4 and h = 1,
    and forward top levels that skip the zero quarters of half-zero sources (two of the three
    forwards per step). At 2^19 a forward costs 32.9 instead of 34.2 cycles per vector; the
    products gain only at h = 4. `judge.py bench`, 21 rounds: `lc-amd` 0.9933, `lc-intel` 0.9865.
  - Tried for the products, not kept: their bottom (forward, windows, leaf products, inverse) as
    one list-scheduled asm statement per group, 18 knob variants: tiles 52.9-55.0 against 53.0
    cycles per vector for the intrinsics.
  - Merged as #170. CI: inv 0.9881 (Xeon 8573C 0.9847, EPYC 7763 0.9939, 0.9857); all 6
    problems 0.9922 (composition 1.0224).
  - Submitted the merged `main.cpp` (#170): [409369](https://judge.yosupo.jp/submission/409369)
    AC 18 ms from one launch spike (near_262144_02 18 ms, peers 6; `tools/spikes.py`: clean 11);
    the same file again: [409370](https://judge.yosupo.jp/submission/409370) AC 11 ms, 11.5 MiB.
- Next (round 2): the inverse takes ~7.1 ms of ~12 (`lc-amd`); the floor (read, write) is ~5 ms.
  Its product bottoms (~30 cycles per vector, the multiply pipes' bound ~20) are the largest part;
  asm scheduling did not help. Smaller: the inverse top level of the first product and the forward
  top level of the second as one pass (~0.3-0.5%, guess); exp and sqrt have the same pattern.
- 2026-10-10, claude (round 3, issue #62; owner lane of lib/poly, no lib change).
  - Kept, problem-local:
    - `io::read_bulk` and `RUN_EARLY` (`lib/run/early.hpp`, same code as the old copy):
      `judge.py bench` (`lc-bench`, 31 rounds) 0.9889.
    - `step.hpp`, in-process A/B against main's lib (`lc-bench`, the problem's computation at
      N = 500000, median ratios): the fused top pass 0.9947-0.9953 (warm, 21-31 rounds), steps to
      2^18 0.9947, last step 0.9991; with the direct half 0.9939 (fresh arena per call, 61 rounds;
      the fused pass alone 0.9967). The copy cost ~80 us per MiB.
  - Whole process, round 3 against main, `judge.py bench`, slowest 3 cases: `lc-bench` 11.72 ->
    11.61 ms (0.9882, 41 rounds); `lc-intel` 11.71 -> 11.66 (0.9936, 21 rounds; `read_bulk` is
    `Reader::read` there).
  - Checks: 25/25 official tests (`lc-amd`); `stress.py` 400 rounds; ASan/UBSan on all 25 official
    cases, file and pipe input, outputs equal.
  - Product bottom, L1 ablation (`lc-bench`, 64 groups, cycles per vector): full 30.7-31.5; leaf
    products and inverse butterflies 20.2; forward butterflies and windows 10.6; leaf products
    alone 15.1; forward butterflies, windows and leaf products 25.4. The parts add up almost
    linearly. Tile and b at 32 offsets (0-3968 B): 30.3-30.7, no aliasing effect.
  - Zen 3 probes (`lc-bench`, independent ops): strict multiply/ALU alternation (M A, M N, L A)
    3.05 ops per cycle, but M M A A, M A A, M A M A A 4.0; vector loads (256-bit or broadcasts) at
    most 2 per cycle in all; a 256-bit load across a 64-byte line takes 2 slots, across a 32-byte
    boundary 1.
  - Tried, not kept:
    - Bottom in two phases per chunk of groups (all window fills, then all leaf products): 1.022
      (chunks of 4), 1.025 (16) on the whole computation.
    - Bottom in three passes per tile (asm `forward_bottom`, windows and leaf products,
      asm `inverse_bottom`): last step 0.968-0.987 at 2^17-2^20 in the problem's arena, but
      1.013-1.017 with the same buffers in another arena, steps to 2^18 1.0035; with a fresh arena
      per call 0.9957 against 0.9967 for the fused pass alone. Placement decides; neutral on average.
    - Leaf product with each window slice loaded once (9 loads instead of 16 memory operands):
      15.4 against 15.0 cycles per leaf, bottom 31.4 against 30.7. Loads are not its limit.
    - Leaf product with two accumulators per parity (add chains of 3): 14.8 against 14.9 per leaf,
      bottom 30.1 against 30.65 (-1.7% in L1, ~0.5% of the inverse, estimate). A lib change for
      every product user; not made.
  - Not pursued (counted): Newton without truncation (g~ = g - x^k e g) needs transforms of 4k;
    T(g_2k) from the second product's transform needs the truncated part's forward (no saving);
    F at length 2L from F at L needs a fresh half (no saving). The step stays 5 transforms.
- Next: the inverse is ~7.2 ms of ~11.6; output `write()` ~1.7 ms and start/exit ~1.4 ms are the
  floor. Product bottoms run at ~2.5 vector ops per cycle (bounds: 4 ops, 2 multiplies, 2 loads per
  cycle); what holds them back is not found (not loads, not placement, not alternation).
