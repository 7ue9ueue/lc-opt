# mul_modp_convolution

c_k = sum over i j = k (mod P) of a_i b_j mod 998244353, P prime, 2 <= P <= 524288 (so at most
524287 = 2^19 - 1). 5 s.

Best judged: ours, 11 ms: [409660](https://judge.yosupo.jp/submission/409660) (`main.cpp` of #311).
Before: 12 ms, [409289](https://judge.yosupo.jp/submission/409289) (#135) and 409350.
Record when opened (issue #35): 45 ms.

## Design

- Primitive root g (trial division of P - 1). For i, j != 0, i = g^x: the nonzero outputs are the
  cyclic convolution of length n = P - 1 of A[x] = a_(g^x), B[x] = b_(g^x). One linear product of
  length 2n - 1 <= 2^20 (`lib/ntt`), folded: c_(g^k) = d_k + d_(k+n).
  c_0 = a_0 (b_0 + sum b) + b_0 sum a; the sums come out of the fold pass. P = 2 directly.
- Transform: for 2^lg >= 512 (P >= 131), `ntt::Product` (`lib/ntt/product.hpp`: convolution_mod's
  radix-8 first level, bottom stage and fused inverse top). Other P: `ntt::Convolution`.
- Slots: g^(x + n/2) = P - g^x, so x and x + n/2 meet indices s and P - s, slot
  s = min(g^x, P - g^x) in [1, n/2]. Input and output keep each slot's two indices side by side:
  one random access serves x and x + n/2 (half of round 2's).
- Input: a_1.. and b_1.. are parsed into the factors' lower halves, then folded into quads
  [a_s, b_s, a_(P-s), b_(P-s)] (16 bytes; the second pair read backwards). The quads live in the
  factors' upper halves (which `ntt::Product` does not read): slots 1..2^lg/8 in a's, the rest in
  b's. No memory beyond the product's layout.
- Gather: powers g^x by a vector Shoup multiply (8 lanes, step g^8); per lane one 16-byte load of
  the slot's quad, a 4x8 transpose, and blends by y > (P - 1)/2 give A[x], B[x], A[x + n/2],
  B[x + n/2].
- Output: scatter c_s + 2^32 c_(P-s) into slot s of b's buffer (one 8-byte store per slot), unfold
  into a's buffer (even dwords forward, odd dwords backward), `../convolution_mod/fields.hpp`,
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
- Next (round 2): the product (4.46 ms) is `lib/ntt`'s. Gather and scatter (0.78 + 0.66) resisted
  prefetch, fusion and partitioning. No idea left outside `lib/` worth a round (guess).
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  12.04 → 11.93 ms (0.993). 40/40 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib, issue #156 round 2): `Product` is now a wrapper of `ntt::Product`
  (`lib/ntt/product.hpp`, moved from `../convolution_mod`); the copied radix-8 level is gone.
  New in the product: convolution_mod's bottom stage (two groups per asm statement, no weight
  array) and fused inverse top; P in [131, 2^17] also takes it (radix-4 top at odd lg).
  `judge.py bench`, `lc-bench` (EPYC 7B13), 31 rounds, slowest 3 cases: 12.18 -> 11.91 ms,
  ratio 0.9764. 40/40 official tests, `stress.py` 300 rounds and 24 known large cases,
  ASan/UBSan on 7 official cases (file and pipe input).
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).

- 2026-10-10, claude (round 3). `lc-amd` and `lc-bench` (EPYC 7B13), judge flags. Exploration
  files: `lc-opt-explore/mul_modp_convolution/` (`probe.py` stamps convolve(), `phases.sh`,
  `variants.py`, `split_bench.cpp`, `asan.sh`). Phases: in-process medians of 21 interleaved runs
  on p_max_00 (us): parse a + b, interleave or fold, gather, multiply, scatter, output.
  - Kept, slots (Design): main 1694 / 255 / 681 / 4314 / 476 / 1975 (9395); slots 1694 / 248 /
    412 / 4312 / 313 / 2067 (9046). The gather and scatter make half the random accesses; the
    output pays ~0.1 ms for the unfold.
  - Kept details: input sums in the fold pass instead of the gather (gather 475 -> 412, fold +4).
    Unfold in one pass, then `fields::write`: equal to unfolding per 25600-value block into a
    staging buffer (`judge.py bench` 10.71 vs 10.70 ms) and simpler.
  - Lost (same phases): the unfold fused into the formatter (fields.hpp's `divide` and `store`
    with loads from the folded layout, four 16-byte loads and two `shufps` per 16 values): output
    2129 vs 2090 us for the staging buffer. Gather with `vpgatherdq` 666 vs 490; two groups of 8
    slots per iteration 551 vs 490; 64-bit addresses through the stack 517 vs 475 (GCC turns the
    index store and reloads into `vpextrd`); scatter of 16 slots per iteration 350 vs 344;
    scatter's store loop unrolled 307 vs 313 (noise).
  - Measured, not pursued: x^n - 1 = (x^(n/2) - 1)(x^(n/2) + 1) splits the product into two
    linear products of length n - 1 <= 2^19 (inputs A_lo +- A_hi from the gather, c from the
    scatter). Two `ntt::Product` of 2^19 against one of 2^20, fresh mappings, medians of 41:
    4552 vs 4594 us (-1%; lg 19 takes the radix-4 top level). The quads would span four upper
    halves. Not worth it.
  - The binary always needs libstdc++ (the driver links it without `--as-needed`; checked with
    `readelf -d`), so its loading (~0.4 ms, lib/run/notes.md) cannot be avoided from source.
  - `judge.py bench`, `lc-bench`, slowest 3 cases (p_max_00, p_max_01, large_05): 31 rounds
    11.09 -> 10.71 ms (0.968); final version, 41 rounds, a busier VM: 11.56 -> 11.17 (0.962).
  - Checks: 40/40 official tests (`lc-amd`); `stress.py` 500 rounds and 24 known large cases,
    200 rounds built with `-march=x86-64-v3`; ASan/UBSan on all 40 official cases, file and pipe
    input.
  - `lc-k68` (Linux 6.8, judge proxy), static builds, `tools/runner.c`, 31 rounds, slowest of
    p_max_00, p_max_01, large_05: 11.30 -> 10.87 ms (0.962). Every huge page of the product's
    mapping is written before it is read (parse, fold, tables, text), so 6.8's split of the huge
    zero page (convolution_mod_large round 4) does not apply.
  - CI (#311, merged): Xeon 6973P-C 0.9989, EPYC 9V45 0.9843, Xeon 8573C 0.9576 (geomean 0.980).
  - Submitted the merged `main.cpp` (#311), 2 of 5 this session:
    [409659](https://judge.yosupo.jp/submission/409659) AC 14 ms (spike on large_04; `spikes.py`:
    clean 11), [409660](https://judge.yosupo.jp/submission/409660) AC 11 ms, 15.9 MiB. Best judged
    12 -> 11 ms.
- Next: left outside `lib/`: fold 0.25, gather 0.41, scatter 0.31 and unfold ~0.1 ms of ~9.0 in
  process. The product (4.3 ms), the parse (1.7) and `write()` (~1.7) belong to `lib/ntt`,
  `lib/io` and the kernel. Halving the random accesses again needs a dense index of the cosets of
  a subgroup larger than {1, -1}; none is cheap (guess).

## Sources

- Discrete log reduction of multiplicative convolution mod a prime: standard (Rader-style index
  map). No code read. Slots: g^((P - 1)/2) = -1 for a primitive root g (standard).
- `lib/ntt` (`ntt::Product`), `lib/io`; `../convolution_mod/fields.hpp` (shared).
