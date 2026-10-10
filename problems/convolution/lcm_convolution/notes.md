# lcm_convolution

N <= 10^6, a_i, b_i < 998244353; print c_k = sum over lcm(i, j) = k of a_i b_j for k = 1..N. 5 s.
Large tests: N = 10^6 (max_random), near primes and near prime squares; ~20 MB input, 10 MB output.

Record when opened: 37 ms. Best judged: ours, [409237](https://judge.yosupo.jp/submission/409237),
16 ms.

## Design

- c = Moebius(zeta(a) zeta(b)), zeta = sums over divisors. Per prime p: zeta x_ip += x_i (i
  ascending), Moebius x_ip -= x_i (i descending). The passes commute.
- Same skeleton as `../gcd_convolution` (read for ideas, code written here): pairs [a | b R] so one
  load fetches both, b pre-scaled by R = 2^32 for one Montgomery reduction per product; one
  mapping in 2 MiB pages (pairs, later c as dwords; then b, later the rough sweeps' prefix
  copies; then the output text).
- a is parsed into the second half of the pair array, so the interleave runs upwards (pair k
  overwrites only a values below k). Zeta 3 is fused with it, target-contiguous: 24 targets per
  step take 8 sources k / 3 from below (permute + blend).
- Zeta 5..13: scalar passes. Zeta 2 + product + Moebius 2 in one ascending sweep:
  A_k += A_k/2, c_k = A_k B_k - A_k/2 B_k/2 (the second product recomputed), 8 targets per step.
- Moebius 3..13: scalar passes on c.
- Primes >= 17: two stages. Stage 1: m > 1 with all prime factors in 17..97; stage 2: m > 1 with
  all prime factors >= 101. Rough m with factors on both sides come from composing them:
  1.77 N contributions per transform instead of 2.14 N for all rough m. The lists are built at
  compile time (constexpr sieve over the 30030 wheel's indices; stage 2 has 120759 m <= 10^6).
  - Zeta = stage 2 after stage 1. Stage 2's sources are below n / 101; a copy of that prefix
    with stage 1 applied serves them, so both stages share one walk. Target segments of 2^15
    pairs descend: stage 1 sources (below the segment) are still old. Segment 0 by source.
  - Moebius = stage 1 after stage 2, each c_im -= final c_i. A prefix copy gets stage 2 first;
    segments of 2^16 dwords ascend, stage 1 reads final values below.
  - In a segment, m <= 4096 go by m over a run of sources; larger m by source i <= n / 4097 over
    a run of m (carried index; sentinels 0 and 2^20 frame the list). m <= 512 take the segment in
    32 KiB pieces (zeta: ascending pieces, stateless bounds).
- Runs from `.preinit_array`, ends with `_exit`. Output: `../convolution_mod/fields.hpp`.

## Log

- 2026-10-09, claude, round 1. `lc-amd` (EPYC 7B13), judge image and flags.
  - v1 (this `main.cpp`): 29/29 official tests, slowest 16.4 ms (`judge.py test`). Stress
    300 rounds (`stress.py`: brute N <= 3000, per-prime reference up to N = 300000). ASan/UBSan
    (GCC 15.2, -O1): 29/29 tests plus pipe input.
  - Phases, max_random_00, medians of 21 (ms): parse 3.53, interleave + zeta 3 0.48, zeta
    5..13 0.63, rough zeta 1.99, zeta 2 + product + Moebius 2 0.60, Moebius 3..13 0.56, rough
    Moebius 1.96, output 3.75. Sum 13.5; compute 6.2.
  - The rough sweeps: by m 0.95 ms (~0.92 N contributions), by source 1.15 ms (~1.1 N), so
    ~3.5 cycles per contribution. `perf` on `lc-intel`: the samples sit on the target loads.
  - No gain (all probe medians, same run as a v1 control):
    - L1 sub-blocks (16 KiB) for dense generators (m <= 64..256, sources <= 16..64) inside each
      L2 segment, stateless bounds: rough zeta 2.09-2.36 vs 1.99; rough Moebius 1.82-1.95 vs 1.96.
    - Segments of 2^13, 2^14, 2^16 pairs (zeta) / 2^14, 2^15, 2^17 dwords: 2^15 / 2^16 best.
    - `prefetchw` 4..32 spokes ahead in the by-source loops: +0.25 ms each sweep.
    - Keeping P in a register (GCC reloaded the constant per contribution): rough zeta 1.98 vs
      1.99. A `cmov` modular subtract for c: rough Moebius 2.29 vs 1.95.
    - Zeta 5..13 fused into the interleave by inclusion-exclusion (F_k = x_k + sum over
      squarefree d of -mu(d) F_k/d; d = 3 by vectors, the rest by source per 1536-pair block), and
      Moebius 3..13 likewise fused into the product sweep (products written as dwords over the
      dead pairs, c into its own array). Small-prime total 2.23-2.27 ms vs 2.26 for any number of
      fused primes: the extra terms (Σ 1/d = 1.15 N for all five primes vs 0.84 N) cost what the
      saved sweeps did.
    - Not storing the pairs above n / 2 in the product sweep: 0.596 vs 0.603 ms.
  - PR #59 merged (CI: 29/29 on 3 runners, slowest 16.5-18.0 ms). Submitted its `main.cpp`:
    [409219](https://judge.yosupo.jp/submission/409219), AC, 17 ms, 22.4 MiB (1/5).

- 2026-10-09, claude, round 2. `lc-amd`, judge image and flags. Phases: in-process rdtsc
  (3.05 GHz TSC), max_random_00, medians of 21-31 runs, base = round 1 in the same run.
  - Contributions per transform (counted in Python, N = 10^6): all rough m 2.144 N. Two stages
    with the cut at T: T = 100 1.767, 200 1.786, 500 1.858, 1000 1.932. Three stages (50, 1009):
    1.716 (not built). Squarefree-only stage lists (zeta by Moebius-style recursion): 0.569 +
    1.174, no saving.
  - v2a: two stages, lists built at run time from a wheel-index sieve: build 0.36 ms, rough zeta
    +0.29, Moebius -0.30; `judge.py bench` 1.018 (slower). The descending zeta loop then had two
    conditions per step.
  - v2b: lists at compile time (constexpr; GCC's limits are 2^25 ops per constant and 2^18
    iterations per loop, so the sieve runs over wheel indices, not values; compile 6 s, binary
    537 KB), sentinel 0 below the large list: rough zeta 1.96 -> 1.91 ms, Moebius 1.98 -> 1.69.
    Split per piece: zeta stage 1 0.64, stage 2 1.18; Moebius stage 1 0.53, stage 2 1.02.
  - L1 pieces (as gcd v7) for tiny m, bound x piece: Moebius 512 x 8192 dwords: 1.67 -> 1.53;
    256 / 1024 / 2048 and 4096-dword pieces within 0.02. Zeta with descending pieces: +0.04 to
    +0.34 for every bound (128..4096) and piece (2048, 4096), stage 2 alone too. Zeta with
    ascending pieces and stateless bounds: -0.05 to -0.06 (256..1024 the same). Kept 512.
  - Zeta large loop ascending (scan down for the begin, then add upwards): +0.23. Dropped.
  - Split m by-m / by-source: 1024 / 2048 / 4096 / 8192 / 16384 / 32768: rough sum 3.59 / 3.45
    / 3.37 / 3.37 / 3.40 / 3.51 ms (two runs). Kept 4096. Segments 2^14/2^15, 2^16/2^17 (zeta pairs /
    Moebius dwords): worse than 2^15/2^16.
  - c now overwrites the pairs (c_k over pair k/2, read for the last time at target k), so b's
    space holds the prefix copies: product sweep unchanged (0.59 ms).
  - v2 (this `main.cpp`): rough zeta 1.96 -> 1.80 ms, rough Moebius 1.97 -> 1.51; in-process
    total 9.18 -> 8.53. `judge.py bench`, 31 rounds, slowest 3 cases: 16.89 -> 16.29 ms (0.966).
  - Checks: 29/29 official tests (`judge.py test`); `stress.py` 300 rounds (gcc:15.2.0 on
    `lc-intel`); ASan/UBSan (-O1, x86-64-v3) on all 29 tests plus pipe input, tokens equal.
  - PR #97 merged. CI ratios (EPYC 7763, 3 runs): 0.971, 0.960, 0.958.
- 2026-10-09, claude: submitted the PR #97 `main.cpp` (v2),
  [409237](https://judge.yosupo.jp/submission/409237): AC, 16 ms, 22.4 MiB (2/5 for the
  problem). Was 17 ms.
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  16.37 → 16.05 ms (0.981). 29/29 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib/io #21, round 3): submitted the #176 `main.cpp`,
  [409381](https://judge.yosupo.jp/submission/409381): AC 16 ms, 22.5 MiB; clean 16 ms
  (`tools/spikes.py`). Best judged stays 16 ms.
- 2026-10-10, claude (lib, issue #156 round 2): the huge-page allocation comes from
  `lib/mem/huge.hpp` instead of a local copy. Same stripped executable as before (judge flags,
  `lc-amd`).
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).

## Next

- Rough sweeps 3.3 ms: zeta costs 1.2x Moebius per contribution, cause unknown (pairs are 8 B;
  descending order measured slower everywhere it was tried). Three stages save 3% more
  contributions.
- I/O (parse 2.9 ms, format + write 0.66 ms in process) belongs to `lib/io` and the formatter.

## Sources

- Design from `../gcd_convolution/notes.md` (zeta/Moebius by per-prime passes, wheel sweep,
  Montgomery pairs); dual direction worked out here.
- `.preinit_array` start: `../convolution_mod/solution.cpp`.
- L1 pieces for tiny multipliers: `../gcd_convolution/notes.md` (v7).
