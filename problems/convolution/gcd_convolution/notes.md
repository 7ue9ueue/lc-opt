# gcd_convolution

N <= 10^6, a_i, b_i < 998244353; print c_k = sum over gcd(i, j) = k of a_i b_j for k = 1..N. 5 s.
Large tests: N = 10^6, 999982..999984, 994008..994010 (997^2 - 1 + {0, 1, 2}), random values;
19.8 MB of input, 10 MB of output.

Record when opened: 37 ms (407011). Best judged: ours, [409380](https://judge.yosupo.jp/submission/409380),
14 ms (#176). Earlier: 15 ms, [409214](https://judge.yosupo.jp/submission/409214).

## Design

- c = Moebius(zeta(a) zeta(b)), where zeta sums over multiples. Each is a product of commuting
  per-prime passes, so the passes can go in any order and be grouped.
- a and b interleaved as 64-bit pairs [a, b R mod P]: one load fetches both. b is pre-scaled by
  R = 2^32 so one Montgomery reduction gives a b mod P.
- Memory: one mapping in 2 MiB pages (12 MB): the pairs, with a parsed into their first half
  (the interleave runs downwards, so it only overwrites a values already read), then b, later c,
  then the stage-2 sums of the rough sweeps (in the last huge page, already touched by b).
- Primes 2..13: one AVX2 pass each. Zeta 3 is fused with the interleave; zeta 2 with the product
  and Moebius 2 (c_i = A_i B_i - A_2i B_2i, the second product recomputed).
- Primes >= 17 together ("rough": coprime to 30030), in two stages (as `../lcm_convolution`):
  stage 1 takes m > 1 made of primes 17..97, stage 2 m > 1 made of primes above 100. Composing
  them covers every rough m: 1.77 N terms per transform instead of 2.14 N. Both lists are built
  at compile time (sieve over the wheel's indices, 5760 spokes); stage 2 has 120759 m <= 10^6.
  - Stage 2 changes only i <= N / 101, inside segment 0, so its terms go to separate sums (the
    dual of lcm's prefix copy). Zeta = stage 1 after stage 2: sources are read segment by segment
    upwards (each before it changes) for both stages; the stage-2 sums t2 then get stage 1 on
    themselves and are added to the pairs. Moebius = stage 2 after stage 1: c_i = C_i - sum of
    final c_im with segments downwards, stage-2 terms into b2; segment 0 goes last by target,
    stage 1, then stage 2.
  - Multipliers go by m over a run of targets up to 16384 (stage 1) and 2048 (stage 2); larger
    ones by target (i <= N / 2049) into 64-bit sums.
  - m < 256 take their sources by L1-sized pieces (32 KiB) of each segment, so one fetch of a
    source line from L2 serves all of them. Stage-2 targets i <= 8 walk each piece first: their
    walks bring the piece into L1, and the tiny m then find it there.
- Output: `../convolution_mod/fields.hpp` (10-byte fields, 16 values per step); the text buffer
  is the first 250 KB of the pair array, dead by then.
- The input mapping is advised `MADV_SEQUENTIAL` (as `../bitwise_and_convolution`): its `munmap`
  skips marking pages accessed.
- Runs from `.preinit_array` and ends with `_exit` (as `convolution_mod`).

## Log

- 2026-10-09, claude, round 1. `lc-amd` (EPYC 7B13), judge flags, `tools/judge.py bench`, 21
  rounds, slowest 3 cases, median ms:
  - v0: scalar per-prime passes on `std::vector`s, byte sieve: 26.6 (`judge.py test`: 25.5).
    In-process phases: allocation 3.3, parse 3.0, sieve 2.8, zeta 5.4, product 0.9, Moebius 2.7,
    output 4.3.
  - v1: pairs, AVX2 small-prime passes, fused passes above, huge pages, all primes p^2 > N in one
    pass by target: 18.6 (0.70 of v0).
  - v2: one 12 MB mapping instead of five (13 huge pages zeroed before, 6 now), segmented sieve
    (2.8 -> 0.4 ms): 17.3 (0.927 of v1).
  - v4: one sweep for all primes >= 17, no sieve: 16.2 (0.956 of v2).
  - v5: scalar loads instead of `vpgatherdd` for strided dwords: 16.1 (same as v4 within noise;
    kept, simpler). `judge.py test`: 13.6-14.9 ms on the large cases.
  - I/O floor (read a and b into the same mapping, print a): 11.3. So 4.8 ms of compute remain.
- In-process phases of v5 (ms): parse a 1.5, parse b 1.75, interleave + zeta 3 0.5, zeta 5..13
  0.58, rough zeta 1.5, zeta 2 + product + Moebius 2 0.49, Moebius 3..13 0.45, rough Moebius 1.43,
  output 4.5, exit 0.24; start ~1.0 before `main`.
- Why the rough sweep is not faster: per-prime passes for p >= 17 touch 1.51 N lines from L3 at
  ~1 ns each (~3 cycles; at L3 bandwidth). The sweep has 2.14 N contributions (composite m add
  0.63 N), each a separate line from L2, and Zen 3 fills L1 from L2 at 32 B/cycle: ~2 cycles each.
  Measured: 2.2 TSC cycles per contribution by m, 2.5 by target. Net 0.1 ms per transform, plus
  no sieve.
- Sweep tuning: segment 2^13..2^16 pairs: 2^15 best (smaller: more per-segment overhead; larger:
  no gain). Split 1024 / 2048 / 4096: 1.58 / 1.50 / 1.50 ms rough zeta. Carrying the next index
  per target and multiplier, and floor division by reciprocal multiply (no `div`): 1.58 -> 1.50.
- Counted, not built: splitting the rough sweep into stages by prime size cuts contributions
  (17..1000 and > 1000: 1.93 N; three stages: 1.72 N) but adds a pass over the array per stage.
- Page faults on `lc-amd` in Docker: 0.10 ms per 2 MiB page with MADV_HUGEPAGE, 0.86-0.91 ms per
  2 MiB in 4 KiB pages.
- Harness: `judge.py bench` reruns write the same output file; truncating 10 MB of tmpfs adds
  ~1-2 ms per run compared with `judge.py test`. Same for every source.
- `lc-intel`: the first `judge.py test` run took 194 ms on one case, later runs 14-17 ms. A guess:
  THP compaction on fault (defrag = madvise, 13.7 GB of page cache, 0.45 GB free).
- 2026-10-09, claude: submitted the PR #43 `main.cpp` (v5).
  - [409189](https://judge.yosupo.jp/submission/409189): AC, 24 ms (1/5). Large cases 15-16 ms,
    except near_prime_squared_00 at 24 ms.
  - [409190](https://judge.yosupo.jp/submission/409190), same file: AC, 17 ms (2/5). Large cases
    15-17 ms. First; next is 37 ms (407011).
- 2026-10-09, claude, round 2. `lc-amd`, judge image and flags. "Probe": whole-process wall time
  on max_random_01 in the judge's Docker image, binaries interleaved, 41-61 runs, medians (ms).
  - v6a: `.preinit_array` start, `_exit`, `fields.hpp` formatter with its buffer in the pair
    array: probe 13.87 -> 13.17 (v5 -> v6a). `judge.py bench`, 11 rounds: 16.07 -> 15.39 (0.953).
  - Parse b in chunks along the rough zeta sweep (a in the pairs' second half; each segment
    interleaved just before the sweep reads it; zeta 3 then a plain pass): probe 13.26-13.29 vs
    v6a 13.12-13.17. Chunks of 2^16 / 2^17 / 2^18 / 2^20 (one read) values: 13.33 / 13.29 /
    13.23 / 13.15. Each extra `Reader::read` call costs more than the cache reuse gains. Dropped.
  - v7: L1 pieces for tiny rough multipliers (above). Probe, tiny bound x piece (pairs):
    64 x 1024/2048/4096: 13.10/13.03/13.06; 128 x same: 12.99/12.99/12.94; 256 x same:
    13.16/13.03/12.93; 512 x 4096/8192: 12.94/13.03; 8192-pair pieces are worse for every bound
    (13.03-13.09). v6a in the same runs 13.21-13.25. Kept 256 x 4096 (Moebius: 8192 dwords).
  - `judge.py bench`, 31 rounds, slowest 3 cases: v5 16.28, v6a 15.41 (0.937), v7 15.17 (0.931).
  - Checks: 29/29 official tests; `stress.py` 300 rounds; ASan/UBSan (-O1, x86-64-v3) on all 29
    official tests, file and pipe input, tokens equal to the expected output.
  - PR #56 merged. CI ratios: EPYC 7763 0.933, EPYC 9V45 0.936 and 0.932 (geomean 0.933).
- 2026-10-09, claude: submitted the PR #56 `main.cpp` (v7),
  [409214](https://judge.yosupo.jp/submission/409214): AC, 15 ms, 22.4 MiB (3/5). Was 17 ms.
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  15.13 → 14.86 ms (0.979). 29/29 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib/io #21, round 3): submitted the #176 `main.cpp`,
  [409380](https://judge.yosupo.jp/submission/409380): AC 14 ms, 22.5 MiB; clean 14 ms
  (`tools/spikes.py`: spikes only on small cases). New best judged (was 15 ms).
- 2026-10-10, claude (lib, issue #156 round 2): the huge-page allocation comes from
  `lib/mem/huge.hpp` instead of a local copy. Same stripped executable as before (judge flags,
  `lc-amd`).
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).
- 2026-10-10, claude, round 3 (`agent/gcd_convolution-r3`). Judge image and flags; timing on
  `lc-bench` (EPYC 7B13), builds and tests on `lc-amd`. Phases: in-process `CLOCK_MONOTONIC`
  stamps on max_random_01, medians of 21 runs, ms; "wall" is fork to exit with the output unlinked
  first. Sources, scripts, raw numbers: `lc-opt-explore/gcd_convolution/`.
  - Main's phases: wall 13.54; parse 3.26, interleave + zeta 3 0.49, zeta 5..13 0.57, rough zeta
    1.35, zeta 2 + product 0.48, Moebius 3..13 0.33, rough Moebius 1.27, output 3.89, input
    unmap 0.66; start + exit 1.2.
  - Terms per transform (`count3.py`): rough m > 2048 (by target) hold 1.106 N of 2.144 N. Two
    stages cut at 100 leave 1.767 N: tiny 0.640, by m 0.428, by target 0.699.
  - v1, two stages: rough zeta 1.38 -> 1.25, rough Moebius 1.28 -> 1.04; wall 0.973 of main.
    Per part (rdtsc): zeta tiny 0.30, by m 0.28, stage-1 by target 0.09, stage-2 by target 0.52;
    Moebius 0.25, 0.25, 0.05, 0.41. Stage-2 by target costs ~2.1 TSC cycles per term for i <= 32
    (one L2 line each) and 2.9 for i > 32 (loop overhead per target and segment).
  - Stage-1 split (by m up to) 2048 / 4096 / 8192 / 16384 / 32768: rough zeta 1.247 / 1.239 /
    1.212 / 1.210 / 1.214, Moebius 1.037 / 1.022 / 1.016 / 1.018 / 1.011. Kept 16384.
  - Stage-2 targets i <= T by L1 pieces: after the tiny m, zeta +0.02 to +0.22 (T = 8..64),
    Moebius -0.03. Before the tiny m: zeta 1.228 -> 1.163, Moebius 1.018 -> 0.966 (T = 8);
    T = 4..16 within 0.02; 2048-pair pieces worse. Kept T = 8, before.
  - `MADV_SEQUENTIAL` on the input: unmap 0.651 -> 0.617. Never unmapping (the Reader in static
    storage): the cost moves to exit, wall unchanged. Kept the advice.
  - No gain: vector loops running past a piece's end (no scalar tail), within noise; tiny bound
    192 / 384 / 512, +0.01 to +0.02; zeta segments 2^14 (+0.1) or 2^16 (same); Moebius 2^15 (same).
  - Three stages (cuts 60 and 300, 1.667 N), written generically over the stage count with run-time
    stage views: rough zeta 1.36 vs 1.25 for the same code with two stages (669 by-m multipliers
    per segment instead of 533), and that generic two-stage code was itself 0.09 slower in zeta
    than the templated one. Dropped. Four stages (40, 100, 1000): the constexpr tables did not
    build (a guess: GCC's constexpr operation limit).
  - Final: rough zeta 1.38 -> 1.16, rough Moebius 1.27 -> 0.97 (phases). `judge.py bench`,
    slowest 3 cases: 31 rounds 14.81 -> 14.25 ms (0.955); an earlier run of 21 rounds 14.62 ->
    14.24 (0.977).
  - Checks: 29/29 official tests; `stress.py` 300 rounds, now with segment-edge sizes (32767 ..
    131072) against the reference; ASan/UBSan (-O1, x86-64-v3) on all 29 tests, file and pipe
    input, tokens equal. Compile ~6 s (constexpr lists); text 516 KB (was 97 KB).

## Next

- The rest is I/O: parse 3.3 ms (`lib/io`), output 3.9 ms (`write()` ~3.3), input unmap 0.62,
  start + exit 1.2. Compute ~3.9 ms: rough sweeps 2.1, small primes and product 1.8 (L3-bound
  passes over 8 MB).
- Rough sweeps run at ~2 cycles per term where each term is its own L2 line; more stages cut
  terms (three: -0.1 N) but each stage adds by-m calls per segment, which cost more here.
- `MADV_SEQUENTIAL` on the input is now in three problems (bitwise_and, bitwise_xor, gcd): a
  candidate for `lib/io`.

## Sources

- Two stages and their compile-time lists: `../lcm_convolution` (its round 2; the `Stages` and
  `Multipliers` construction is taken from there, the gcd direction worked out here).
- `MADV_SEQUENTIAL` on the input: `../bitwise_and_convolution/solution.cpp`.
