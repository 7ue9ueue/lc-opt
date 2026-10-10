# prefix_sum_of_polynomial

f of N <= 2^19 coefficients mod 998244353; print the N + 1 coefficients of g with
g(n) = sum_(i < n) f(i). 5 s. Slowest tests: max_random_00/01 (N = 2^19), then random_00/01
(N > 2^18, the same transform lengths). small (N <= 100), medium (N ~ 10^4), examples.

Best judged: ours, 13 ms: [409571](https://judge.yosupo.jp/submission/409571) (`main.cpp` of #282),
clean (`tools/spikes.py`).
Record when opened (issue #89): 143 ms.

## Design

- Faulhaber: with y / (e^y - 1) = sum_j B_j y^j / j!, g_0 = 0 and
  g_(k+1) = (1 / (k+1)!) sum_(t >= k) a_t t! B_(t-k) / (t-k)!. Every input needs B_0 .. B_(N-1)
  (f = x^(N-1) prints them), so one division for the Bernoulli numbers is unavoidable.
- Half length: y / (e^y - 1) = E(y^2 / 4) - y / 2 with E(w) = sqrt(w) coth(sqrt(w)) = C(w) / S(w),
  C_j = 1 / (2j)!, S_j = 1 / (2j + 1)!. With abar_t = a_t t! / 2^t:
  g_(k+1) = 2^k / (k+1)! sum_j abar_(k+2j) E_j - a_(k+1) / 2. E has K = ceil(N / 2) terms (K a
  power of two >= 128), and the sums split by the parity of k into two products with E of
  length 2K: abar's even and odd terms reversed at [0, K), the sums in the lower half.
- E = C / S in 4 blocks of b = K / 4: sigma = 1 / S mod w^b (`lib/poly/inverse.hpp`), then
  Q_j = sigma (C_j - R_j) mod w^b with R_j = sum_(t <= j) (W_t Q_(j-t))[b, 2b) over windows
  W_t = S[(t-1) b, (t+1) b), from stored transforms of length 2b (`inverse_product_sum`); the
  scheme of `lib/poly/division.hpp`.
- Passes in 32 lanes (`lib/poly/factorials.hpp` chains): 1 / i! for i < 2K, split into C and S by
  a permute and two 128-bit stores; abar_t (multipliers (t + 1) / 2) stored reversed and split into
  the two product inputs; output z_k = 2^k / (k+1)! (multipliers (k + 1) / 2, backward) times the
  interleaved sums, minus a_(k+1) / 2, in place of the input.
- Memory: one `poly::Arena`; a (input, then output) and three zones of 2K words reused across
  phases: x (S, T(Q_0..2), block buffer, then the even product), y (inverse scratch, T(sigma),
  T(W_1..3), then the odd product), z (C turned into E in place, sigma, then T(E)). 5 huge pages
  at N = 2^19 (9 in v1).
- `lib/io` input (`io::read_bulk`); output in 10-byte fields
  (`problems/convolution/convolution_mod/fields.hpp`, judge-specific); `RUN_EARLY` start.

## Floor

`lc-bench`, whole process, judge flags, max_random_00/01 and random_00/01 (scratch `timeit.py`:
judge.py's build and runner without the checker): read N + 1 numbers with `io::read_bulk`, write
N + 1 values with `fields.hpp`, nothing else (scratch `floor.cpp`): 4.58 ms median of the slowest
case (21 rounds).

## Log

- 2026-10-10, claude (round 1): first solution. Timing on `lc-bench` (EPYC 7B13, judge flags),
  tests on `lc-amd`. Scratch files: `lc-opt-explore/prefix_sum_of_polynomial/`.
  - v1: E = C / S by sigma = 1 / S mod w^(K/2) and one Karp-Markstein step at length K; separate
    buffers (9 huge pages). Phases at max_random_00 (ms, `CLOCK_MONOTONIC` probe): read 0.9,
    1 / i! 0.25, inverse 1.7, T(sigma) 0.3, q0 0.75, T(q0) 0.3, residual 0.75, q1 0.8, weights
    0.35, T(E) 0.6, products 1.55 + 1.55, output 0.45, write 0.34.
  - v2: buffers in three reused zones (5 huge pages). A bug found by the official tests: the
    reused zones left stale words in [0, K - 16 C) of the product inputs for N with 16 C < K;
    cleared now.
  - v3: E by 4-block division (above): inverse 0.82, T(sigma) 0.12, windows 0.41, blocks 2.85:
    E 4.2 ms against 4.6. Blocks of K / 2 (2 blocks, the Karp-Markstein step) 1.027, of K / 8
    (8 blocks, 7-term residual sums) 1.028 against 4 blocks (`timeit.py`, 21 rounds).
  - `timeit.py` 21 rounds, slowest of max_random_00/01, random_00/01: floor 4.58, v1 14.94,
    v2 14.22, v4 (v3 cleaned up) 13.75 ms. Above the floor (9.2 ms): transforms 7.9 (E 4.2,
    products 3.7), passes 1.0.
  - Checks: 26/26 official tests (`judge.py test`, `lc-amd`); `stress.py` 400 rounds (`lc-amd`):
    N <= 1500 against `brute.cpp` (values by Horner, Newton's forward differences, expanded),
    larger N up to 2^19 by g(x + 1) - g(x) = f(x) and g(0) = 0 at random x; coefficients random,
    in {0, 1, P - 1}, or zero; ASan/UBSan on all 26 official cases, file and pipe input.
  - Merged as #282 (new problem: CI checks only).
  - Submitted the merged `main.cpp` once (1 of 5): [409571](https://judge.yosupo.jp/submission/409571)
    AC 13 ms, 15.0 MiB; `spikes.py`: clean 13 ms, P(clean run) = 0.77. Not resubmitted.
- Next:
  - The two products (3.7 ms) and E (4.2 ms) are 86% of the time above the floor. E is input-
    independent; a faster reciprocal (Harvey's 13/9 M(n)) or a cheaper blockwise division would
    cut its 0.82 ms inverse and 2.85 ms of blocks.
  - T(E) at length 2K (0.6 ms): half of it is T_K(E), whose blocks' transforms of length K / 2
    are known; the twisted half is not.
  - Memory: computing E before reading the input and folding a_(k+1) / 2 into the products could
    save a zone (~0.2 ms of page faults, a guess).

## Sources

- Faulhaber's formula with Bernoulli numbers, and y / (e^y - 1) = (y / 2) coth(y / 2) - y / 2:
  standard; derived here, no code read.
- Bernoulli numbers by one power series division of the even series: R. P. Brent, D. Harvey, "Fast
  computation of Bernoulli, Tangent and Secant numbers", Computational and Analytical Mathematics
  (2013) (the idea; no code read).
- Division step: A. Karp, P. Markstein, "High-precision division and square root", ACM TOMS 23
  (1997) (v1, v2). Blockwise division: our `lib/poly/division.hpp` (#81), rewritten here for a
  power series quotient.
- Transforms, inverse: our `lib/poly`; chains: `lib/poly/factorials.hpp`.
