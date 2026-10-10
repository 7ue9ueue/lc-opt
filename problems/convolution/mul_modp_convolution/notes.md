# mul_modp_convolution

c_k = sum over i j = k (mod P) of a_i b_j mod 998244353, P prime, 2 <= P <= 524288 (so at most
524287 = 2^19 - 1). 5 s.

Best judged: ours, 12 ms: [409289](https://judge.yosupo.jp/submission/409289) (current `main.cpp`, #135).
Record when opened (issue #35): 45 ms.

## Design

- Primitive root g (trial division of P - 1). For i, j != 0, i = g^x: the nonzero outputs are the
  cyclic convolution of length n = P - 1 of A[x] = a_(g^x), B[x] = b_(g^x). One linear product of
  length 2n - 1 <= 2^20 (`lib/ntt`), folded: c_(g^k) = d_k + d_(k+n).
  c_0 = a_0 (b_0 + sum b) + b_0 sum a; the sums come out of the gather.
- Transform: for 2^lg = 2 * 4^j >= 256 (all P > 2^17), `Product`: the radix-8 first level of
  `../convolution_mod` (copied), then `lib/ntt`'s recursion. Other P: `ntt::Convolution`.
- Input: a_1.. and b_1.. are parsed into the factors' lower halves, then interleaved into pairs
  a_i + 2^32 b_i (one 8-byte load serves both factors). The pairs live in the factors' upper
  halves (which the radix-8 level does not read): pairs 1..2^lg/4 in a's, the rest in b's. No
  memory beyond `ntt::Convolution`'s layout.
- Gather: powers g^x, g^(x+16) by a vector Shoup multiply (16 lanes, step g^16), lane order
  0 1 4 5 2 3 6 7 so that two `vpgatherdq` and two `shufps` give A and B in natural order.
- Output: scatter c[g^k] into b's buffer, `../convolution_mod/fields.hpp`,
  `.preinit_array` start, `_exit`.

## Log

- 2026-10-09, claude (round 1). `lc-amd`, judge flags. Phases: in-process `CLOCK_MONOTONIC`
  medians of 15 runs on p_max_00 (P = 524287), output to a file.
  - Floor (`../floor.py --fixed`, 11 rounds, 3 largest cases): 6.58 ms.
  - v1: gather `a[g^x]`, `b[g^x]` with `vpgatherdd` from two arrays, then the product.
    40/40 official tests, slowest 13.3 ms. Phases (ms): parse 1.6, gather 1.44, product 4.45,
    scatter 0.66, format + write 2.0.
  - v2: gather fused into the radix-8 level (no stored permuted factors; inputs in a separate
    4 MiB). Gather + level 1.83, product 4.20: +0.12 ms in-process; bench ratio 1.032. Dropped.
  - v3/v4: v2 with interleaved pairs. Gather + level 1.15 (the loads alone: 0.58 ms; without
    loads 0.61). Interleave: chunked parse of b 0.78 ms extra (v3); full parse then AVX2
    interleave 0.31 (v4). In-process -0.25 ms, but `judge.py bench` 1.009-1.019 against v1
    (3 runs). A fork/exec harness agreed with the bench only with tmpfs output (v4 12.25 vs v1
    12.20 ms; with disk output v4 12.10 vs 12.39). Guess: the extra 4 MiB (2 huge pages) slows
    the later tmpfs page allocations. Dropped.
  - Gather variants on v4 (in-process, load phase): scalar 8-byte loads 1.16 vs `vpgatherdq`
    1.17; software prefetch 4 or 16 iterations ahead 1.30, 1.45; scalar + prefetch 8: 1.35.
    Scatter with `prefetchw` 2, 8, 32 iterations ahead: 0.67-0.70 vs 0.66. All dropped.
  - v5 (kept): v1 with pairs in the factors' upper halves (above). Gather 1.44 -> 0.78, parse +
    interleave 1.87, product 4.46. `judge.py bench`, 41 rounds, slowest 3 cases: v1 12.41,
    v5 12.06 ms (ratio 0.9765).
  - Checks: 40/40 official tests (slowest 12.0 ms); `stress.py` 300 rounds (all primes < 300,
    then random primes < 3000, both product paths) plus 24 known cases at P = 524287, 65537,
    262147 and a random prime; ASan/UBSan on all 40 official cases, file and pipe input.
  - Submitted the merged `main.cpp` (#135): [409288](https://judge.yosupo.jp/submission/409288)
    AC 22 ms (judge jitter), resubmitted: [409289](https://judge.yosupo.jp/submission/409289)
    AC 12 ms, 15.5 MiB.
- 2026-10-09, claude (round 2). `lc-amd`, judge flags. In-process medians of 21 interleaved runs
  on p_max_00, phases (ms): gather, product + scatter, output. Baseline (main): 0.78, 5.13, 2.07.
  - Last inverse layer (`scale_radix2`) fused into the scatter: one Shoup multiply per output on
    the folded brackets. 40/40 tests. 0.77, 5.18, 2.07; `judge.py bench` (41 rounds, 3 slowest)
    12.30 vs 12.41 ms, ratio 1.013. The scatter is store-bound; the saved 8 MiB pass is in L3.
    Dropped.
  - Scatter partitioned by output block (2^14 values; entries value | index << 32 in b's buffer),
    then per block: scatter in L1/L2, format, write(2). 40/40 tests. 0.78, 5.92, 2.34. Dropped.
  - Same partition by slice of c (2^14, 2^16, 2^17 values) into a prefaulted 4 MiB, then a
    sequential scatter, original output: product + scatter 6.32-6.34 against 5.14. The partition
    pass (scalar bucket tails, 8-byte entries) alone costs more than the random scatter. Dropped.
- 2026-10-10, claude (issue #156): the local `fields.hpp` copy is gone; the solution includes
  `../convolution_mod/fields.hpp` (it was byte-identical). `main.cpp` changes in one comment line; the
  judge's command builds byte-identical executables from main's and this `main.cpp` (`lc-amd`).
- 2026-10-10, audit (claude): [409350](https://judge.yosupo.jp/submission/409350) (2026-10-10
  01:54 UTC) was not logged before; who submitted it is not recorded. Current `main.cpp` (#163):
  AC 12 ms, 15.8 MiB, no spike on the slowest case. Ties best judged (409289).
- Next: the product (4.46 ms) is `lib/ntt`'s. Gather and scatter (0.78 + 0.66) resisted
  prefetch, fusion and partitioning. No idea left outside `lib/` worth a round (guess).
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  12.04 → 11.93 ms (0.993). 40/40 official tests. ASan/UBSan on the 3 largest cases, file and pipe.

## Sources

- Discrete log reduction of multiplicative convolution mod a prime: standard (Rader-style index
  map). No code read.
- `lib/ntt`, `lib/io`; `../convolution_mod` for the radix-8 level and `fields.hpp` (shared).
