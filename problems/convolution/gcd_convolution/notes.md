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
  (the interleave runs downwards, so it only overwrites a values already read), then b, later c.
- Primes 2..13: one AVX2 pass each. Zeta 3 is fused with the interleave; zeta 2 with the product
  and Moebius 2 (c_i = A_i B_i - A_2i B_2i, the second product recomputed).
- Primes >= 17 together: x_i += sum of x_im over m coprime to 30030 ("rough", m > 1), from a
  wheel (5760 spokes, rank table; both constexpr). Zeta takes the sources segment by segment
  upwards (each source is read before it changes), Moebius uses c_i = C_i - sum of final c_im with
  segments downwards; segment 0 goes by target. Multipliers m <= 2048 go by m over a run of
  targets; larger ones by target (i <= N / 2049) into 64-bit sums. No sieve is needed.
- Rough multipliers m < 256 take their sources by L1-sized pieces (32 KiB) of each segment, so
  one fetch of a source line from L2 serves all of them; larger m go over the whole segment.
- Output: `../convolution_mod/fields.hpp` (10-byte fields, 16 values per step); the text buffer
  is the first 250 KB of the pair array, dead by then.
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

## Next

- The rest is I/O: parse 3.3 ms (`lib/io`, #21), `write()` ~3.4 ms. Compute left ~4.5 ms, of
  which the rough sweeps ~2.6 ms.
