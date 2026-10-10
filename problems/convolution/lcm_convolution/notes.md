# lcm_convolution

N <= 10^6, a_i, b_i < 998244353; print c_k = sum over lcm(i, j) = k of a_i b_j for k = 1..N. 5 s.
Large tests: N = 10^6 (max_random), near primes and near prime squares; ~20 MB input, 10 MB output.

Record when opened: 37 ms. Best judged: ours, [409551](https://judge.yosupo.jp/submission/409551),
14 ms. Earlier: 16 ms, [409237](https://judge.yosupo.jp/submission/409237).

## Design

- c = Moebius(zeta(a) zeta(b)), zeta = sums over divisors. Per prime p: zeta x_ip += x_i (i
  ascending), Moebius x_ip -= x_i (i descending). The passes commute.
- Same skeleton as `../gcd_convolution` (read for ideas, code written here): pairs [a | b R] so one
  load fetches both, b pre-scaled by R = 2^32 for one Montgomery reduction per product; one
  mapping in 2 MiB pages: the pairs (later c as dwords), then 1 MB of scratch (chunks of b, later
  the sweeps' prefix copies, later the output text). 5 huge pages.
- a is parsed into the second half of the pair array, so the interleave runs upwards (pair k
  overwrites only a values below k). b is parsed in chunks of 262080 values into the scratch,
  each interleaved right away. Zeta 3 is fused with the interleave, target-contiguous: 24 targets
  per step take 8 sources k / 3 from below (permute + blend).
- Zeta 2 + product + Moebius 2 in one ascending sweep: A_k += A_k/2, c_k = A_k B_k - A_k/2 B_k/2
  (the second product recomputed), 16 targets per step.
- Moebius 3..13: scalar passes on c.
- All other zeta primes in one sweep, three stages. Stage 0: m > 1 made of 5, 7, 11, 13 (222 m,
  0.74 N contributions; the old-source form needs every such m). Stage 1: m > 1 with all prime
  factors in 17..97; stage 2: m > 1 with all prime factors >= 101. Rough m with factors on both
  sides come from composing 1 and 2: 1.77 N contributions per transform instead of 2.14 N. The
  lists are built at compile time (constexpr sieve over the 30030 wheel's indices; stage 2 has
  120759 m <= 10^6, 487 KB).
  - Zeta = 0, then 1, then 2. Stage 1's sources are below n / 17, stage 2's below n / 101:
    prefix copies with the earlier stages applied serve them (prefix 1 by per-prime passes,
    prefix 2 by source). Target segments of 2^15 pairs descend: stage 0 sources (below the
    segment) are still old. Segment 0 by source, stage 0 first.
  - Moebius = stage 1 after stage 2, each c_im -= final c_i. A prefix copy gets stage 2 first;
    segments of 2^15 dwords ascend, stage 1 reads final values below.
  - In a segment, m <= 4096 go by m over a run of sources; larger m by source i <= n / 4097 over
    a run of m (carried index; sentinels 0 and 2^20 frame the list). m <= 256 of all stages take
    the segment together in 32 KiB pieces (zeta: ascending pieces, stateless bounds).
  - Each update is 2 loads, 3 vector ops, 1 store: runs of one m go by pointers, 4 per step, the
    modulus held in a register (GCC reloaded it per update).
- The input mapping is advised `MADV_SEQUENTIAL` (as `../bitwise_and_convolution`).
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
- 2026-10-10, claude, round 3 (`agent/lcm_convolution-r3`). Builds on `lc-amd`, timing on
  `lc-bench` (EPYC 7B13), judge image and flags, max_random_00 on tmpfs, output unlinked before
  each run. "Wall": fork to exit, variants interleaved, median of paired ratios; phases:
  in-process rdtsc. lc-bench ran ~1.4x slow for one scan (noisy neighbour); ratios still hold.
  Sources, scripts, raw results: `lc-opt-explore/lcm_convolution/r3/`.
  - Base phases (ms, 21 runs): parse 3.28, interleave + zeta 3 0.47, zeta 5..13 0.63, rough zeta
    1.85, zeta 2 + product 0.60, Moebius 3..13 0.55, rough Moebius 1.52, output 4.05; in process
    12.97, wall 14.85.
  - Sweep updates: GCC emitted ~12 macro-ops per contribution (32-bit index copies, a reload of
    the constant P). Runs by pointers, 2 per step, P held by an empty asm: rough zeta segments
    1.75 → 1.50, rough Moebius 1.43 → 1.27, Moebius 3..13 0.55 → 0.46; wall 0.964. 4 per step
    and tiny bound 256 (was 512): sweeps -0.06 ms more. No pieces at all: +0.04 zeta, +0.14
    Moebius. Tiny bound 128 / 256 / 512 per stage, pieces 1024..8192: within 0.03 ms.
  - Microbenchmark (`xb.cpp`, one 2^15-pair segment, m prime in a range, TSC cycles per
    contribution): m 17..97: 2.81 / 2.29 / 2.00 / 1.81 / 1.85 / 2.09 / 2.36 with pieces of
    256 / 512 / ... / 8192 pairs / the whole segment; m 101..509: best 2.15 (8192); m 1000..4096
    over the segment 2.77. Fit: ~1.6 per update in L1, ~9 per run.
  - Zeta 5..13 as stage 0 of the sweep (old sources, every 5..13-smooth m: 0.738 N, 94% from 17
    m <= 245) instead of 4 passes over the pairs: zeta 5..13 + rough zeta 2.27 → 2.11 ms; wall
    0.992. Stage 0's tiny updates cost 0.67 ns each.
  - Tiny m of all stages per piece (one load of the piece): 0.84 → 0.80 ms (zeta), 0.33 → 0.32
    (Moebius).
  - Product sweep, 16 targets per step (the 8 source products in one vector): 0.598 → 0.552 ms.
  - b parsed in chunks into a 1 MB scratch that later holds the prefixes and the text: 7 → 5
    huge pages (minflt 469 → 467); parse b + interleave 2.07 → 1.99 ms; wall 0.988. Chunks of
    65520: the prefixes then fault a new page (+0.06 ms); 32760: more parser calls (+0.12).
  - `MADV_SEQUENTIAL` on the input, as `../bitwise_and_convolution` (-0.04 ms there).
  - Phases now: parse a 1.68, parse b + interleave 1.99, zeta sweep 2.06, product 0.55, Moebius
    3..13 0.48, Moebius sweep 1.32, output 4.06.
  - `.rodata`: 512 KB more constants, read once at the start: +8 page faults, wall +0.10 ms
    (1.0073, IQR 1.0024-1.0131, 41 runs). The stage-2 list (487 KB) costs about that. uint8
    gaps (120 KB) would save ~6 faults but add a load and an add to each of its 1.2 M updates
    per run: not built.
  - Counted, not built (Python, N = 10^6): three stages (cuts 100, 1000) 1.716 N per transform
    against 1.767 N; four (50, 300, 1000) 1.654 N. Moebius 5..13 as a sweep stage needs final
    sources, so every 5..13-smooth m (0.738 N) against 0.511 N in passes: break-even estimate.
  - Segments again, with the new loops (wall, 31-41 runs): Moebius 2^15 dwords 0.988 and 0.992
    (IQR 0.987-0.997; targets and sources now fit L2), 2^14 0.999, 2^17 0.998; zeta 2^14 pairs
    1.008, 2^16 0.999 (2^16 with Moebius 2^15: 1.000). Kept Moebius 2^15, zeta 2^15.
  - `judge.py bench`, `lc-bench`, 31 rounds, slowest 3 cases: 15.21 → 14.29 ms (0.944) before
    the Moebius segment change; 15.16 → 14.14 ms (0.936) after.
  - Checks (final): 29/29 official tests (`judge.py test`, slowest 15.2 ms on `lc-amd`);
    `stress.py` 300 rounds (gcc:15.2.0); ASan/UBSan (-O1, x86-64-v3) on all 29 tests, file and
    pipe input, tokens equal to the expected output.
  - PR #265 merged. CI: geomean 0.947 (EPYC 7763 0.932, EPYC 9V74 0.952 and 0.959).
- 2026-10-10, claude: submitted the PR #265 `main.cpp`.
  - [409550](https://judge.yosupo.jp/submission/409550): AC 22 ms, 20.5 MiB; clean 14 ms
    (`tools/spikes.py`: max_random_01 22 against 14 in its peers). 1/5 for this version.
  - [409551](https://judge.yosupo.jp/submission/409551), same file: AC 14 ms, 20.5 MiB, no
    spike. Large cases 13-14 ms. New best judged (was 16 ms). 2/5.

## Next

- Compute is ~4.9 ms of ~14 ms: zeta sweep 2.06, Moebius sweep 1.32 (~0.8 ns per update; L1
  updates ~0.5 ns plus ~3 ns per run). Three or four stages: -0.05 to -0.11 N per transform.
- I/O (parse 3.2 ms, output 4.1 ms in process) belongs to `lib/io` and the formatter.

## Sources

- Design from `../gcd_convolution/notes.md` (zeta/Moebius by per-prime passes, wheel sweep,
  Montgomery pairs); dual direction worked out here.
- `.preinit_array` start: `../convolution_mod/solution.cpp`.
- L1 pieces for tiny multipliers: `../gcd_convolution/notes.md` (v7).
- `MADV_SEQUENTIAL` on the input: `../bitwise_and_convolution/solution.cpp`.
