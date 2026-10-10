# shift_of_sampling_points_of_polynomial

f(0), ..., f(N - 1) of f with deg f < N, and c mod 998244353; print f(c + i) for i < M.
N, M <= 2^19. 5 s. Slowest tests: max_random_00..03 (N = M = 2^19, random c), then
type1_random_00/03 (N ~ 4 10^5, outputs partly at sample points). type0 (all outputs are
samples), type1 (c < N, outputs run past the samples), type2 (no overlap), type3 (c + M wraps
past P, outputs may hit the samples again), c_0, N_1.

Best judged: none yet.
Record when opened (issue #80): 41 ms.

## Design

- Lagrange on 0, ..., N - 1, for c + k outside the samples mod P:
  f(c + k) = (-1)^k P_k sum_i g_i h_(k + N - 1 - i), g_i = f(i) / (i! (N - 1 - i)!),
  h_t = (-1)^t / (d + t), d = c - N + 1, P_k = (d + k) ... (d + k + N - 1). A middle product:
  one cyclic convolution of length L = 2^lg >= 2 max(N, M) (lg even) with g at [0, N) and h_t at
  L/2 - N + 1 + t, so the sums sit at L/2 + k, the upper half in natural order. Transform as
  multipoint_evaluation_on_geometric_sequence (#76): `forward_radix8` for g, a full radix-8 pass
  for h, `ntt::detail::Subtrees`, the halves' top groups, the last radix-2 level in the output.
- Sample points: outputs with c + k = i mod P, i < N, are f(i), copied. The rest form at most two
  ranges (before the samples, from c; after them, from c' = N), each one product, and in each the
  d + t are nonzero. Only type3 cases with a short overlap need two products (type3_random_02).
- Passes, all in 32 lanes (`lib/poly/factorials.hpp`, new this round):
  - weights: a reversed chain of 1 / i! 2^32 (lane tops from the factorial table) stored to a
    scratch in b, then pairs from both ends: w_i = w_(N-1-i) = s_i s_(N-1-i), one product for
    two positions, g in place over the input.
  - prefix products: G_t = prod -(d + u) over the lane, into b (h's place) and, for t >= N, into
    y (G_(N+k)). Negated factors put the (-1)^t of h into the chain (lane lengths even).
  - inverses: a reversed chain R_t = 2^32 / G_(t+1) from the inverted lane totals; h_t = G_t R_t
    in place; z_k = R_(k-1) G_(N+k) kappa in place of y, kappa = (-1)^N A_s' / A_s (L/8)^-1 2^96
    for the lanes s of k - 1 and s' of N + k (A: prefix products of the lane totals), blended
    per vector at the one lane boundary of N + k inside a lane of k - 1.
  - output: f(c + k) = (u - w)[k] z_k / 2^32; z_0 from the lanes' constants (scalar).
- Padding: the scans cover 32 C >= T = N + M - 1 positions. Padding goes at the end, or at the
  front when d + t = 0 there (a range that ends at P - 1); then position T starts a virtual lane
  32 and G_T = 1. Stores to y are clipped to the range, so ranges and copied samples share it.
- `lib/io` input straight into a; output in 10-byte fields
  (`problems/convolution/convolution_mod/fields.hpp`, judge-specific). One `mem::Arena` mapping;
  `RUN_EARLY` start.

## Floor

`lc-bench`, whole process, judge flags, max_random_00..03 and type1_random_00/03 (scratch
`timeit.py`: judge.py's build and runner without the checker): read N + 3 numbers with
`io::read_bulk`, write M values with `fields.hpp`, nothing else (scratch `floor.cpp`): 4.28 ms
median of the slowest case (15 rounds); 4.37 in the run next to v4 below.

## Log

- 2026-10-10, claude (round 1): first solution. Timing on `lc-bench` (EPYC 7B13, judge flags).
  Scratch files: `lc-opt-explore/shift_of_sampling_points_of_polynomial/`.
  - First promoted #79's factorial machinery to `lib/poly/factorials.hpp` (see
    `lib/poly/notes.md`, Factorials); polynomial_taylor_shift uses it (bench 1.0002, noise).
  - v1: the design above with weights in two passes (chain with f / i! in place and s stored;
    then f s_(N-1-i)). Bugs found by the official tests: the type3 range ending at P - 1 put
    d + t = 0 in the padding (fixed by front padding) and needed G_T past the last lane (virtual
    lane 32). The first multi-range version read f into a vector and ran each range in its own
    mapping: type1_random_00 11.6 ms against 10.6 for max_random; ranges now share the mapping
    and y (9.7 ms).
  - Phases at max_random_00 (ms, `CLOCK_MONOTONIC` probe, medians): read 0.98, weights 0.69,
    prefix 0.53, inverses 0.86, radix-8 passes 0.12 + 0.14, halves 1.92 + 1.93, top groups
    0.05 + 0.07, output 0.18, write 0.34. Warm (passes run twice): weights 0.48, prefix 0.46,
    inverses 0.84: the passes are compute-bound; page faults ~0.2 ms in weights.
  - v2: `store_clipped` always inlined (GCC called it 32 times per block, spilling the chains):
    prefix 0.449 -> 0.438, inverses 0.858 -> 0.849, output 0.153 -> 0.125 ms (warm pass
    microbenchmark, `mkbench.py`).
  - Lost, v3: kappa in the output pass instead of the inverses pass (z = R G there):
    inverses 0.83 -> 0.71, output 0.12 -> 0.24; `judge.py bench` 15 rounds: 1.0049.
  - Lane layout check (`lanes_bench.cpp`): a chain pass over 2^20 positions with 32 long lanes
    (C = 32784) costs 0.29 ms (store) / 0.47 (load, product, store); blocks of short lanes
    0.24-0.30 / 0.43-0.54. Powers of two alias (C = 32768: 1.41 ms). Long lanes kept.
  - Y copy in the prefix pass costs 0.08 ms (0.427 -> 0.350 without it).
  - v4 (kept): weights as one chain pass storing s, then pairs from both ends (2.5 products per
    position instead of 3, one pass over a less): weights 0.425 -> 0.357 ms warm;
    `judge.py bench` 21 rounds, max_random_00..02: v2 10.52, v4 10.47 ms, ratio 0.9970.
  - Phases of v4 at max_random_00 (ms, last 3 of 7 runs): read 0.89, weights 0.54, prefix 0.42,
    inverses 0.81, radix-8 0.26, halves 3.86, top groups 0.12, output 0.12, write 0.34. Above
    the floor (6.5 ms): transform 4.2, passes 1.9. Some cold runs showed prefix at 1.2-4.7 ms
    (huge page faults on `lc-bench`; not seen in the warm runs).
  - `timeit.py` 15 rounds, max_random_00..03 and type1_random_00/03: floor 4.37, v4 10.86 ms
    (slowest case per round; max_random cases 10.5 each).
  - v5 (kept): the scans' lambdas capture local copies by value. `__m256i` stores may alias
    anything, so members (`h()`, `lane_start()`, `y_`) and captured references were reloaded
    after every store. Warm passes: weights 0.354 -> 0.339, prefix 0.44 -> 0.36, inverses 0.84
    -> 0.77, output 0.124 -> 0.116 ms. Whole process (cold) unchanged: `judge.py bench` 21 rounds
    0.9993; cold probes show the passes bound by first-touch faults there (prefix 0.42-0.46 ms,
    with runs at 0.8-6 ms on huge page faults).
  - Aligned h (delta rounded to 8, wrong output, timing only): prefix 0.43 -> 0.37 ms warm, the
    other passes unchanged. In the max case delta = 1 is forced (g fills the lower half, the
    outputs the upper half); g at an offset would need the full radix-8 pass for a.
  - 8 or 16 lanes instead of 32 (`lanes_count_bench.cpp`, one chain over 2^20): forward 0.48 /
    0.37 / 0.31 ms, backward 0.56 / 0.49 / 0.47: 32 kept.
  - Checks: 32/32 official tests (`judge.py test`, `lc-amd`); `stress.py` 400 rounds (`lc-amd`):
    N, M <= 1500 against `brute.cpp` (Newton differences), larger by Lagrange at 6 points; c at
    0, inside the samples, just past them, wrapping past P, ending at P - 1 +- 2, random;
    ASan/UBSan on all 32 official cases, file and pipe input.
- Next:
  - The transform is 4.2 of the 6.5 ms above the floor (shared with #76, #79).
  - inverses (0.77 ms warm): z costs two products per output; kappa is per lane pair because
    the lanes of G are local. Reading G_(N+k) from b where its lane is behind would drop the Y
    copy (0.08 ms) for part of the positions.
  - Cold runs: first touches of b and y cost more than the compute gains above; fewer or
    earlier-touched pages may matter more than products now.

## Sources

- Lagrange interpolation on 0, ..., N - 1 with weights (-1)^(N-1-i) / (i! (N - 1 - i)!), and the
  shift as a convolution with 1 / (x - i): standard; derived here, no code read.
- Middle product: G. Hanrot, M. Quercia, P. Zimmermann, "The middle product algorithm I", AAECC
  14 (2004) (the idea; no code read).
- Batch inversion: P. Montgomery, "Speeding the Pollard and elliptic curve methods of
  factorization", Math. Comp. 48 (1987). Montgomery reduction: P. Montgomery, "Modular
  multiplication without trial division", Math. Comp. 44 (1985).
- Transform: our `lib/ntt` (`product.hpp`); layout from multipoint_evaluation_on_geometric_sequence
  and polynomial_taylor_shift.
