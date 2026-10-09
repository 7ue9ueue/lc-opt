# convolution_mod

N, M <= 2^19 coefficients mod 998244353; print the N + M - 1 coefficients of the product. 5 s.

Best judged: ours, 14 ms: [409184](https://judge.yosupo.jp/submission/409184) (current
`main.cpp`) and [408716](https://judge.yosupo.jp/submission/408716) (the QPoly exploration-011
program, `../SymPoly/work/ntt/yosupo_convolution_mod_large_io_probe.cpp`, a guess from the
submission times and `lib/io/notes.md`). Next other user: 23 ms (393435).

## Design

- `lib/ntt`: one cyclic transform of length 2^lg >= N + M - 1 (2^20 at the maximum).
  For 2^lg = 2 * 4^j >= 256 with both factors at most half the length (all large tests),
  `solution.cpp` replaces the top level: one radix-8 pass per factor reads its lower half once
  and writes the first radix-4 group of both halves (no copy of the lower half, no separate
  pass for the upper half's group); the rest is `lib/ntt`'s recursion and kernels.
- `lib/io` for input (bulk `uint32_t` read straight into the transform buffers).
- Output: `../fixed_width.hpp`, every value in a 10-byte field (judge-specific; the checker
  compares tokens).
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
- Next: the formatter (1.15 ms; ~27 cycles per 8 values against a ~20-cycle port bound, long
  dependency chain per iteration) and the bulk parse (~1 ms besides page faults) are the
  largest parts not fixed by the system.
