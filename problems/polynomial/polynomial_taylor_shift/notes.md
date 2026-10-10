# polynomial_taylor_shift

f of N <= 2^19 coefficients and c mod 998244353; print the N coefficients of f(x + c). 5 s.
Slowest tests: the 12 at N = 2^19 (fft_killer_00..09, max_random_00/01), all about the same.
random_00/01 have N > 2^18 (same transform length, 2^20). c = 0 in medium_c_zero; small N <= 16.

Best judged: ours, 10 ms: [409534](https://judge.yosupo.jp/submission/409534) (`main.cpp` of #251),
clean (`tools/spikes.py`).
Record when opened (issue #79): 33 ms.

## Design

- b_k = (1 / k!) sum_(i >= k) A_i E_(i-k) with A_i = a_i i! and E_j = c^j / j!. One cyclic
  convolution of length L = 2^lg >= 2N: A at [0, N), E reversed (E_j at L/2 - j), so b_k k! is
  coefficient L/2 + k, the upper half in natural order. A is weighted in place over the input,
  and the output needs no reversal.
- E_0 = 1 sits at L/2, in the upper half that `ntt::detail::forward_radix8` assumes zero. Modulo
  x^(L/2) -+ 1 it is +-1 at x^0, which enters the first word of each of the 8 output blocks of the
  radix-8 pass with factor 1: 8 scalar fix-ups after the pass.
- Transform: `ntt::Product`'s layout (`lib/ntt/product.hpp`) as in
  multipoint_evaluation_on_geometric_sequence (#76): `forward_radix8` for both factors (both in
  the lower half), `ntt::detail::Subtrees`, the halves' top inverse groups, and the last radix-2
  level folded into the output pass, which reads (u - w)[k] for the upper half. lg even (2^20 at
  N > 2^17).
- Scans: chains in 32 lanes (4 vectors), lane s over [s C, (s + 1) C), C a multiple of 16 with
  C / 16 odd (cache sets). A step is x <- x (base + u) / 2^32 (Montgomery) with u <- u + step:
  the multipliers are arithmetic progressions (i + 1, j / c, k in Montgomery form), so the
  multiplier costs one add; its odd lanes come from a precomputed odd base (one shift saved).
  Every 8 steps an 8 x 8 transpose per vector puts each lane's terms in position order; the
  chains of block j + 1 are interleaved in source with the stores of block j.
  - weights: input chain i! (forward) and E chain c^j / j! (backward from the lane ends), A_i
    stored in place, E_j at L/2 - j.
  - output: chain z = (L/8)^-1 2^64 / k! (backward), b_k = (u - w)[k] z / 2^32.
  - Lane starts: start(s)! from a table of (1024 k)! and at most 1023 masked chain steps; 1 /
    start(s)! by one batch inversion. The table, the chains and the scans are in
    `lib/poly/factorials.hpp` since #80.
  - Padding positions (n to 32 C <= n + 1023) run without checks: the input is zero there, E's
    terms below L/2 meet A only in the lower half, those below b land in a front padding, and
    outputs past n are not printed.
- N <= 64: Horner's rule in O(N^2). c = 0: the input.
- `lib/io` input (`io::read_bulk`) straight into a; output in 10-byte fixed-width fields
  (`problems/convolution/convolution_mod/fields.hpp`, judge-specific). One `mem::Arena` mapping
  in huge pages (5 at N = 2^19); `RUN_EARLY` start.

## Floor

`lc-bench`, whole process, judge flags, fft_killer_00/01 and max_random_00 (scratch `timeit.py`:
judge.py's build and runner without the checker): read N + 2 numbers with `io::read_bulk`, write
N values with `fields.hpp`, nothing else (scratch `floor_main.cpp`): 4.31 ms median (15 rounds;
4.47 in a busier run; 4.97 on `lc-amd`).

## Log

- 2026-10-10, claude (round 1): first solution. Timing on `lc-bench` (EPYC 7B13, judge flags).
  Scratch files: `lc-opt-explore/polynomial_taylor_shift/`.
  - v1: R = A reversed, E natural, the lower half of the product read reversed; separate
    passes: lane totals of i! (chain), then input and E chains (weights), output chain.
    Phases at N = 2^19 (ms, medians of 9, `CLOCK_MONOTONIC` probe): read 0.95, tables 0.04,
    lane totals 0.083, weights 0.62-0.91, radix-8 passes 0.12 + 0.12, halves 1.93 + 1.94, top
    groups 0.05 + 0.07, output 0.36, write 0.35.
  - v2: the lane loops unrolled and branch-free (GCC had kept the transposes in memory loops
    and per-lane `imul` addressing): weights 0.57-0.79, output 0.37.
  - v3: the layout above (no reversals, A in place, aligned output loads), odd multiplier lanes
    precomputed, chains of the next block interleaved with the stores: weights 0.50-0.54
    (warm 0.40-0.45: ~0.1 ms of it is page faults), output 0.37.
  - Microbenchmarks (warm, min of 30, n = 2^19): a pure chain step is 4.35 cycles per vector
    Montgomery product (lane totals at the 2-pipe bound). Output pass 0.35 ms: chain and
    transposes alone 0.14; with loads from a few streams instead of 64, 0.26 (the 64 read
    streams cost ~0.09); odd lanes of u - w by a shift instead of two more loads: 0.33 -> 0.32.
    Software prefetch of each lane's next line: 0.34 (lost). 16 lanes: weights 0.45, output
    0.34; 8 lanes: weights 0.36, output 0.34 (chain latency-bound); kept 32.
  - v5: lane starts from the factorial table: lane factorials 0.082 -> 0.007 ms. A `constexpr`
    table instead of the generated one costs ~2 s of compile time (2^19 products).
  - Kept (v7, `main.cpp`; v6 plus `lib/mem` and `lib/run`): `judge.py bench`, 21 rounds, slowest
    3 cases: v1 9.79, v7 9.46 ms, ratio 0.970. Floor 4.31. Phases: read 0.87, tables 0.08 (a
    huge page fault), weights 0.50, radix-8 0.24, halves 3.85, top groups 0.12, output 0.34,
    write 0.34; the transform is 4.2 of the 5.15 ms above the floor.
  - Checks: 38/38 official tests (`judge.py test`, `lc-amd` and `lc-intel`); `stress.py` 400
    rounds (`lc-amd`) and 300 (`lc-intel`): N <= 2000 against `brute.cpp`, larger by b(x) =
    f(x + c) at 8 points; c in {0, 1, 2, P - 1} or random; sizes 1-8, 63-66, near powers of
    two, 512 (2m + 1) +- 2 (no partial lane), up to 2^19; ASan/UBSan on all 38 official cases,
    file and pipe input.
  - Merged as #251 (new problem: CI checks only).
  - Submitted the merged `main.cpp` three times (3 of 5 this session; 12 cases at N = 2^19, so
    `spikes.py` gives P(clean run) = 0.45): [409532](https://judge.yosupo.jp/submission/409532)
    AC 18 ms, from a launch spike on fft_killer_02 (18 ms, peers 10; clean 10);
    [409533](https://judge.yosupo.jp/submission/409533) AC 18 ms, spikes on fft_killer_08 (18,
    peers 10) and small_06 (9, peers 0; clean 10); [409534](https://judge.yosupo.jp/submission/409534)
    AC 10 ms, clean: 8 of the 12 large cases 10 ms, the rest 9, random_00 7.
  - Warm microbenchmark (n = 2^19, min of 30): E's stores in the weights pass cost 0.015 ms
    (0.41 -> 0.40 without them), so fusing the E chain into b's radix-8 pass would save little.
- 2026-10-10, claude (issue #80 round, shared code): the factorial table, `Chain`, the
  transposes and the pipelined scan moved to `lib/poly/factorials.hpp` (generator
  `lib/poly/gen_factorials.py`, table to 2^20 + 1023); `factorials.hpp`/`factorials.py` here
  are gone. Same arithmetic; the passes call `poly::detail::scan` with lambdas. `judge.py bench`,
  `lc-bench`, 21 rounds, fft_killer_00..02: main 9.91 ms, lib version 9.97, ratio 1.0002
  (noise); CI: 1.0008 and 1.0050. Cause: the lambdas captured by reference and read members,
  which `__m256i` stores force to reload: weights 0.41 -> 0.50 ms (warm pass microbenchmark,
  `lc-bench`). With local copies captured by value: lane factorials 0.0065 -> 0.0035 (masked
  steps stop at the largest count), weights 0.41 -> 0.42, output 0.33 -> 0.32 ms; `judge.py
  bench` 31 rounds 1.0008. 38/38 official tests, `stress.py` 200 rounds.
- Next:
  - The transform (`ntt::Product`'s) is 82% of the time above the floor.
  - Fuse the E chain into b's radix-8 pass: saves E's stores (0.015 ms measured) and part of
    the pass's reads; likely below noise.
  - Fuse the halves' top inverse groups into the output pass with z precomputed in natural
    order (guess: 0.1 ms, minus the z buffer's fault).
  - 8 lanes for the weights pass only (0.36 against 0.40 ms warm).

## Sources

- Taylor shift as one convolution of factorial-weighted sequences: standard (e.g. the
  convolution form of sum_i a_i C(i, k) c^(i-k)); no code read.
- Transform: our `lib/ntt` (`product.hpp`); layout and output folding from
  `multipoint_evaluation_on_geometric_sequence` and `polynomial_interpolation_on_geometric_sequence`.
- Montgomery reduction: P. Montgomery, "Modular multiplication without trial division",
  Math. Comp. 44 (1985).
