# convolution_mod

N, M <= 2^19 coefficients mod 998244353; print the N + M - 1 coefficients of the product. 5 s.

Best judged: ours, 13 ms: [409226](https://judge.yosupo.jp/submission/409226) (current
`main.cpp`). Earlier versions: 14 ms, [409184](https://judge.yosupo.jp/submission/409184) and [408716](https://judge.yosupo.jp/submission/408716) (the QPoly exploration-011
program, `../SymPoly/work/ntt/yosupo_convolution_mod_large_io_probe.cpp`, a guess from the
submission times and `lib/io/notes.md`). Next other user: 23 ms (393435).

## Design

- `lib/ntt`: one cyclic transform of length 2^lg >= N + M - 1 (2^20 at the maximum).
  For 2^lg = 2 * 4^j >= 256 with both factors at most half the length (all large tests),
  `solution.cpp` replaces the top level: one radix-8 pass per factor reads its lower half once
  and writes the first radix-4 group of both halves (no copy of the lower half, no separate
  pass for the upper half's group); the rest is `lib/ntt`'s recursion and kernels.
- `lib/io` for input (bulk `uint32_t` read straight into the transform buffers).
- Output: `fields.hpp`, every value in a 10-byte field (judge-specific; the checker compares
  tokens), the same bytes as `../fixed_width.hpp`. Per value: w = v / 10 as 8 digits, most
  significant first, in a qword; leading zeros from x ^ (x - 1) and `vpblendvb`; the units digit
  and separator from v - 10w. 16 values per step, ten 16-byte chunks built by `pshufb`; the
  divisions of the next step are issued before the digits of this one. The text buffer sits
  after the NTT tables in their huge page. Shared, not copied: gcd, lcm, mul_mod2n, mul_modp,
  convolution_mod_large and multivariate_convolution_cyclic include it, and six polynomial
  problems.
- The program runs from `.preinit_array` and ends with `_exit`: libstdc++'s initializers
  (iostreams, locales) and exit handlers never run.

## Log
- 2026-10-09, claude: refactored the QPoly program onto `lib/ntt`, `lib/io` input and
  `../fixed_width.hpp` output. 53/53 official tests; stress test 400 rounds (`../stress.py`).
  `lc-amd`, 31 rounds, slowest 3 cases: 14.44 ms vs 14.73 for the 408716 source (QPoly
  `yosupo_convolution_mod_large_io_probe.cpp`), ratio 0.979. Phases at N = M = 2^19 (ms):
  parse 1.9, transform 4.7, output 4.8 (`write()` ~3.7), exit 0.4, start ~1.9.
  Not submitted yet: 2% is about one judge tick (14 ms judged).
- 2026-10-09, claude (round 1). All on `lc-amd`, judge flags, fft_killer_04 unless noted;
  phase times are medians of 21 runs from in-process `CLOCK_MONOTONIC` stamps, wall from fork
  to exit (scratch probe, not committed).
  - Phases of the previous `main.cpp` (ms): start (fork to `main`) 1.02, parse 1.95, tables
    0.09, copy of the lower halves 0.20, half x^n/2 - 1 2.11, half x^n/2 + 1 2.19, scale 0.12,
    format 1.15, `write()` 3.39, exit 0.61. Empty C++ program 1.11 ms, empty C program 0.55:
    loading libstdc++ costs 0.55 ms; its initializers 0.15 (an empty program whose
    `.preinit_array` entry calls `_exit`: 0.96).
  - Radix-8 top, forward only (kept): forward top of a 0.17 (includes a's upper huge page
    fault), of b 0.11, at the memory floor (~6 MiB moved per factor). Replaces the copies
    (0.20) and the upper half's first group (~0.15).
  - Radix-8 inverse with the scale fused (intrinsics): 0.28 ms against 0.24 for the two asm
    groups plus `scale_radix2`; with the scale moved into b's forward pass instead: inverse
    0.22, but b's forward 0.19 (compute-bound). Both lose; the asm passes are kept.
  - `.preinit_array` (kept): start 1.02 -> 0.92 ms, exit -0.05.
  - Interleaved, 41 rounds, wall medians: old 12.87, preinit only 12.71, radix-8 only 12.74,
    both 12.53 (-2.6%). `tools/judge.py bench`, 21 rounds, slowest 3 cases: 14.32 -> 14.06,
    ratio 0.979.
  - Neutral: leaving the input and arrays mapped at `_exit` (12.52, 12.49 vs 12.52);
    unmapping b before the output (`write()` 3.75 vs 3.75 ms in a 10 MiB test).
  - Measured, not pursued: input by 256 KiB `read()` chunks 2.25 ms vs mapped 2.30 (whole
    process, 10 MB file, input touched only); `fallocate` before writing 10 MiB +0.7 ms;
    1 MiB writes +0.4 ms. Formatter: digit count from a float exponent and a `pshufb` table
    0.994 vs 1.013 ns/value in memory (-2%); v / 10^8 through float multiply +9%; unroll 2
    no change.
  - Checks: 53/53 official tests, stress 400 rounds (pipe input), ASan/UBSan on 9 official
    cases (file and pipe input).
  - Interleaving two or three `format8` chains per iteration (to shorten the critical path):
    1.15 and 1.26 ns/value vs 1.02 in memory (spills). Not kept.
  - Submitted the merged `main.cpp` (#40) twice. [409183](https://judge.yosupo.jp/submission/409183):
    AC 22 ms; two outliers (fft_killer_04 22 ms, random_02 15 ms against 6 in 408716), the other
    17 large cases 12-14 ms. [409184](https://judge.yosupo.jp/submission/409184): AC 14 ms;
    large cases median 13 ms (max 14, three at 12), against 14 (max 14) in 408716. Tiny cases
    also took up to 10 ms in both runs: the judge's jitter is several ms, so the maximum over 53
    cases moves by a tick or more between runs.
- 2026-10-09, claude (round 2): new formatter, `fields.hpp`. `lc-amd`, judge flags.
  - Formatter in memory, 2^20 values in 250 KB blocks, min of 40 runs (ms): old
    (`fixed_width.hpp`) 1.04. Steps: 8 MS-first digits per qword with blanks from the lowest set
    bit, units digit and separator from v - 10w, eight 16-byte stores at 10i: 0.92. Divisions of
    the next 8 values issued before this block's digits: 0.79. 16 values per step written as ten
    16-byte chunks (`pshufb`, `por`): 0.73. Blanks by `vpblendvb`: 0.695. Shifts instead of
    multiplies by 2^16 and 2^8: 0.670. High lanes stored by whole 32-byte stores that a later
    store overwrites instead of `vextracti128`: 0.640 (-38%). Exhaustive check: equal to
    `fixed_width.hpp` for every value below P; `write()` equal for 18 counts from 1 to 2^20 - 1.
  - Lost or neutral (ms, same test): store stage before the divisions 0.83; a two-step
    ping-pong unroll 0.737 (base 0.733); chunks interleaved with the digits 0.767 (fewer spills);
    a laundered pointer per constant 0.780 (pointer spills); GCC `schedule-insns` 0.80, with
    `sched-pressure` 0.731, `O3` 0.747; odd lanes from loads 4 bytes on 0.691 vs 0.695;
    multiplies instead of shifts after the store change 0.654-0.665 vs 0.640; 10w by shifts
    0.69 vs 0.67.
  - Why: Zen 3 (`lc-amd`, 3.48 GHz) runs integer vector multiplies on 2 pipes and shifts,
    unpacks and `pshufb` on 2 others (0.5 cycles each, pairs across groups 0.25-0.27);
    `vperm2i128`, `vinserti128` 1 per cycle. Stores per 80 bytes (cycles): 8 xmm at 10i 10.3,
    4 ymm at 20i 5.1, 5 aligned xmm 5.0, 2 ymm + 1 xmm aligned 3.1. The final loop has 99 vector
    ops per 16 values (25 cycles at 4 per cycle) and takes 34: GCC's schedule and 4 spills.
  - Output by mapping stdout instead of `write()` (reopened read-write via `/proc/self/fd/1`,
    `ftruncate`, `MAP_SHARED`): 7.4 ms vs 4.7 for 10 MB on tmpfs, with or without
    `MAP_POPULATE` or `MADV_POPULATE_WRITE`. Not pursued.
  - Text buffer after the tables in Product's last huge page instead of a 250 KB static: whole
    process 0.976 vs 0.979 against main (within noise; kept, no page faults of its own).
    Blocks of 12800, 19200, 25600 values: equal within noise.
  - Transform, measured to see what is left: the 32 subtrees of 2^12 vectors take 3.57 of
    4.5 ms, about 75% of the 4-pipe bound from the kernels' instruction counts; the top levels and
    the L3-resident second level are near their compute bounds. Not changed.
  - Phases now (ms, fft_killer_04, 31 runs): parse 1.91, tables 0.09, top 0.30, transform 4.20,
    format 0.65, `write()` 3.42, unmap 0.39; wall 13.39 (12.89 min).
  - `tools/judge.py bench`, slowest 3 cases: 41 rounds 13.96 -> 13.49 ms, ratio 0.965; an earlier
    run of 31 rounds 0.956.
  - Checks: 53/53 official tests, stress 500 rounds, ASan/UBSan on 11 official cases (file and
    pipe input).
  - Submitted the merged `main.cpp` (#50): [409208](https://judge.yosupo.jp/submission/409208)
    AC 21 ms, from one outlier (max_random_01 21 ms, random_02 16). The 17 large cases have a
    median of 12 ms (409184: 13), min 11, and the rest are at most 13. The judged maximum is jitter;
    best judged stays 14 ms (409184).
- 2026-10-09, audit (claude): the current `main.cpp` (#50) has 7 submissions, over the cap of 5:
  409208 above plus 409220-409224 and 409226, none logged before (not from this session).
  Judged 19, 21, 23, 21, 21, 13 ms. Each slow run has one or two outlier cases (19-23 ms); the
  large cases are otherwise 12-13 ms, matching `lc-amd` (13.49 ms). No further submissions.
- Next: assembly for the formatter loop (34 cycles per 16 values against ~25); the parse
  (lib/io) and the transform kernels (lib/ntt) are the largest parts left in user code.
  `convolution_mod_large` could use `fields.hpp` (its formatter takes ~35 ms of 425).
