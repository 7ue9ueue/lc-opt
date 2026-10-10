# bitwise_xor_convolution

N <= 20; a, b of 2^N values < 998244353; print c_k = sum over i xor j = k of a_i b_j mod P. 5 s.
Input ~20.7 MB (2^21 tokens), output ~10.4 MB.

Best judged: ours, [409428](https://judge.yosupo.jp/submission/409428), 13 ms (lib/io #21 round 5,
PR #198). Round 3 (PR #249): 409530, 13 ms. Round 2: 409210, 14 ms. Round 1: 409198, 15 ms.
Record when opened: 25 ms.

## Design

- c = H (H a . H b) / 2^N, H the Walsh-Hadamard transform: 3 transforms of 20 levels.
- Signed 64-bit lanes, no reduction inside a transform: |x| <= 2^20 P < 2^50.
  Zen 3 runs `vpaddq`/`vpsubq` 4 per cycle; lazily reduced 32-bit lanes need ~3.5 ops per vector
  per level (0.44 per value) against 1 op per 4 values here.
- Rows of 2^12 values (32 KiB of int64), 2^(N-12) rows, rows padded by 64 bytes so a column's rows
  hit different cache sets. Row levels in L1; column levels per strip of one cache line per row.
- The two lane bits: a 4x4 transpose between two radix-4 steps (forward leaves groups of four
  transposed; the product does not care; the inverse transposes back). The inverse (round 3) does
  the 64-bit stage, one level, the 128-bit stage, one level: its values then pack into dwords in
  order with `vshufps` alone (no `vpermq`). It runs a group of four ahead of the reductions.
- Radix-16 passes in inline assembly (`gen_radix16.py`, which simulates its output): 64 add/sub,
  17 stores, one spill, scheduled so that 16 registers are live only at the end.
- x mod P for |x| < 2^51: the bits of x + 0x4338... are the double 1.5 2^52 + x; one FMA with
  1/P and the offset M - M/P rounds the quotient into the low mantissa (error < 0.9, so |r| < 0.9 P).
- a / 2^N mod P (round 2): Montgomery with R = 2^N. k = -a / P mod 2^N (one `vpmuludq`, an
  and), (a + k P) / 2^N lies in (-P, 2P); its bits N to N + 31 are the signed dword.
- Products in doubles (round 2): h = x y, l = fma(x, y, -h) exact, q = round(h / P) by the magic
  constant (|h / P| < 2^51), r = h - q P exact by FMA; r + l, |r + l| < 0.9 P. No reduction of
  y (|y| < 2^20 P) first; 4 FMA-pipe and 4 add-pipe ops per vector, against 7 multiplies before.
  Software-pipelined (round 3): h and M + l of the next row before q, r of this one.
- The input mapping is advised `MADV_SEQUENTIAL` (round 3): its `munmap` skips marking pages
  accessed.
- Order: a's rows while it is parsed, a's columns, a / 2^N kept as dwords (4 MiB); b reuses the
  int64 array: rows, columns with the products, inverse columns; inverse rows, printed per chunk.
- One parse per array (round 4): `progress_read.hpp`, a copy of lib/io's `BulkParser32` that
  calls back after each 256 KiB chunk of text, so each row is widened and transformed as soon as
  its values are parsed, while they are in L2. Each `read_bulk` call ends with shrinking chunks
  and ~1000 tokens parsed one at a time; 2^16-token calls (rounds 1-3) cost 0.25 ms more.
- An array's values are parsed into the last 4 MiB of x and widened in place: row r ends at or
  before the values of row r + 1, and the last row's vectors, 16 values at a time, end before its
  values not yet read. No separate input buffer.
- Output: `../convolution_mod/fields.hpp` per chunk of 2^16 values, so a newline ends every chunk
  (judge-specific; the checker compares tokens). Its text follows the chunk in a (dead by then).
- Memory: 8 MiB + 16 KiB int64 + 4 MiB dwords: 6 huge pages and 4 small ones below them. a's
  dwords are stored strip after strip (8 KiB each, no padding); output chunks reuse a's space.
- Runs from `.preinit_array` and ends with `_exit` (as `convolution_mod`).

## Measurements

AMD EPYC 7B13 (`lc-amd`), GCC 15.2, judge flags, 2026-10-09. Whole process, judge runner,
max_random_00, median of 21 unless noted.

| Program | ms |
|---|---|
| floor: parse a and b only | 6.93 |
| floor: parse, print 2^20 values (fixed width) | 10.19 (11.4-11.8 when the VM was busy) |
| v1 | 18.55 |
| final (`judge.py bench`, 21 rounds, worst of the 3 max cases) | 15.77; v1 19.07, ratio 0.826 |

Floor phases: parse 3.68, print 4.55 (mostly `write(2)` of 10 MB). Final phases (in-process):
rows a 2.72, columns a 0.60, rows b 2.31, columns b 1.04, inverse rows + print 5.34.
Column sub-steps: a transform 0.25, a mod P 0.35 (includes faulting its 4 MiB), b transform 0.27,
products 0.56, inverse 0.25.

Zen 3 (`lc-amd`), dependency-free instructions per cycle: `vpaddq` 4, `vpmuludq` 2, `vpsrlq` 2,
`vaddpd` 2, `vfmadd` 2, `vpunpcklqdq` 2, `vperm2i128` 1.0, `vpermq` 0.79; mixes: mul+add 3.0,
mul+add+add 4.0, mul+srl 4.0, mul+fadd 2.9. 256-bit stores 1 per cycle.
Page faults: 16 MiB in 2 MiB pages 0.73-0.85 ms, in 4 KiB pages 6.7 ms; whole process +0.044 ms
per MiB of huge pages.

## Log

2026-10-09, claude, round 1 (`lc-amd` unless noted; same-run medians):
- v1: int64 lanes, rows of 2^12, 4096-token parses, column pass in strips with a separate product
  loop, output array then `fixed_width::write`. 13/13 tests, 18.55 ms. Phases: parse+rows 7.20,
  columns 2.08, inverse rows 1.02, print 4.58. Parse alone in 4096-token calls: 5.23 vs 3.68.
- v2: parse in 2^16-token chunks; b's last column levels, the products and the first inverse
  levels in one kernel. 18.98 -> 17.62. Rows 7.20 -> 6.10; the fused kernel lost (columns
  2.08 -> 2.24: GCC spills it).
- v3: radix-16 in assembly (slot store at every level); products 4 at a time; fused kernel
  removed. 17.19 -> 16.48. A shift-based scale was wrong (two reductions divide by 2^64);
  caught by the stress test.
- v4: a transformed completely first and kept as dwords; b reuses the int64 array; output per
  chunk. 16.44 -> 16.31 (footprint 17 -> 12.6 MiB).
- v6: transpose loops were not unrolled (`v[4]` on the stack, copy loops) and GCC reassociated
  the radix-4 sums (10 ops for 8): `#pragma GCC unroll` and an empty asm after each level.
  Radix-16 assembly rescheduled ("late spill"). 16.40 -> 15.45.
  Kernels, TSC cycles per vector for 8 bits (two radix-16 passes, L1): GCC radix-16 4.06,
  GCC radix-8 3.00, assembly with a slot per level 3.00, all in place (88 ops) 2.54, late spill
  2.37, late in place (71 ops) 2.38.
- v7: reductions and products stage by stage over 4 vectors. 15.43 -> 15.29. Products in L1:
  6.45 core cycles per vector (7 multiplies; the mul pipes are the limit).
- No gain: rows of 2^11, 2^13, 2^14 (16.1, 16.0, 16.0 vs 15.6); `MADV_POPULATE_WRITE` (15.34 vs
  15.37); a's dwords without `vpermq` (15.39 vs 15.34); 8 products in flight (L1: 6.1 vs 5.6 TSC
  cycles per vector, spills).
- `perf` on `lc-intel` (v4, user cycles): parse 41%, radix passes 15.5%, inverse rows 9%,
  formatting 8.6%, forward rows 8%, products 7.7%, a mod P 4.1%.
- Checks: 13/13 official tests (`lc-amd`, `lc-intel`); `stress.py` 200 rounds against `brute.cpp`
  (N <= 12) plus three N = 20 inputs with known answers (all P - 1; a = P - 1; b a delta), which
  hit the 2^51 bound; ASan/UBSan build on all official tests and 60 random inputs.
- PR #45 merged (CI: 13/13 on 3 runners, slowest 12.7-15.6 ms). Submitted its `main.cpp`:
  [409198](https://judge.yosupo.jp/submission/409198), AC, 15 ms, 23.4 MiB (1/5). Previous
  record 25 ms.

2026-10-09, claude, round 2 (`lc-amd`, judge image and flags). "Probe": whole-process wall time
and in-process `CLOCK_MONOTONIC` phase stamps, max_random_00, programs interleaved, medians of
21-31 runs. Phases of round 1 (ms): start 1.09, parse 3.77, rows 1.26 (incl. 0.4 of x's page
faults), columns a 0.59, columns b 1.05, inverse rows 0.75, print 4.63, exit 0.94; total 14.13.
- `.preinit_array` start: start 1.23 -> 1.12, total ratio 0.990. Kept.
- Products in doubles with a scaled by 2^-N in doubles too: columns b 1.04 -> 0.92, columns a
  0.60 -> 0.70; total ratio 1.001. Scale by Montgomery with R = 2^N instead (5 integer ops, as
  cheap as the old reduction): columns a 0.59, columns b 0.92; ratio 0.990. Kept.
- In-place parse, a stored strip after strip, 6 huge pages instead of 7: rows a 2.74 -> 2.56,
  columns b 0.92 -> 0.85 (a read sequentially); ratio 0.976. Kept.
- Reduction keeps only the low dword (no `vpsubq` of the magic bits), `vinserti128` for two of
  the four 128-bit permutes in the transpose: inverse rows 0.76 -> 0.73; total ratio 1.0005. Kept.
- No gain: row pieces of 64 vectors (lanes and 4 vector bits per 2 KiB piece, then the rest of
  the row; 14.18 vs 14.13); chunks of 2^17, 2^18, 2^20 tokens (ratios 0.997, 1.010, 0.999);
  prefetching the next row in the inverse rows, T0 or T1 (inverse 0.73 -> 0.78).
- Probe totals: round 1 14.13 -> 13.45 ms. `judge.py bench`, 21 rounds, worst of the 3 max
  cases: round 1 15.68, round 2 15.06, ratio 0.960.
- Checks: 13/13 official tests (`lc-intel`); `stress.py` 200 rounds plus the three N = 20
  known-answer cases (judge image on `lc-amd`); ASan/UBSan on all 13 official tests and a pipe.
- Compute left (probe): rows ~0.48 per array, columns a 0.57 (0.18 of it page faults), columns
  b 0.85, inverse rows 0.73. Transforms run ~1.4 instructions per vector-level against a floor of
  ~1.25 for radix-16 in 16 registers; the rest is the I/O floor and process start/exit.
- PR #53 merged. CI ratios: EPYC 7763 0.973, EPYC 9V74 0.963 and 0.931. Submitted its `main.cpp`:
  [409210](https://judge.yosupo.jp/submission/409210), AC, 14 ms, 23.0 MiB (2/5).
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  14.97 → 14.68 ms (0.981). 13/13 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib/io #21, round 3): submitted the #176 `main.cpp`,
  [409379](https://judge.yosupo.jp/submission/409379): AC 14 ms, 23.2 MiB, no spike
  (`tools/spikes.py`). Best judged stays 14 ms.
- 2026-10-10, claude (lib/io #21, round 4): the output text goes after the chunk in a's array
  (dead after the products, already touched), page-aligned, instead of `fixed_width.hpp`'s static
  250 KB (63 page faults), and blocks are 60 pages (`lib/io/notes.md`). `judge.py bench`,
  `lc-amd`, slowest 3 cases: 31 rounds 14.70 → 14.61 ms (0.998); 41 rounds with a copy of main as
  control: 0.991, control 0.995. Outputs byte-identical to main on all 13 tests (judge build,
  ASan/UBSan, pipe input). PR #187 merged; CI 0.9810 (EPYC 9V74 0.981, EPYC 9V45 0.982,
  EPYC 7763 0.980). Not submitted (best judged 14 ms; the gain is ~0.1 ms).
- 2026-10-10, claude (lib/io #21, round 5): output through `../convolution_mod/fields.hpp`
  (in memory 0.64 ms per 2^20 values against 1.04 for `fixed_width.hpp`, now deleted); text
  still page-aligned after the chunk (`../text_buffer.hpp`). `judge.py bench`, `lc-amd`, 31
  rounds, slowest 3 cases: 14.53 → 14.14 ms (0.973). Outputs byte-identical to main on all 13
  tests and 200 random inputs (N 0-13); ASan/UBSan on all 13 tests, file and pipe.
  PR #198 merged; CI 0.9704 (EPYC 7763 0.969, 0.968, 0.975). Submitted
  [409428](https://judge.yosupo.jp/submission/409428): AC 13 ms, 23.0 MiB, no spike
  (`tools/spikes.py`). New best (was 14).
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).

2026-10-10, claude, round 3 (`agent/bitwise_xor_convolution-r3`). Judge image and flags; builds
on `lc-amd`, timing on `lc-bench` (EPYC 7B13, core clock 3.42 GHz, TSC 3.05 GHz). "Probe": phase
stamps in-process plus fork-to-exit wall time, max_random_00, output unlinked before each run.
"Kernel micro": one kernel over N = 20 data laid out as in the program, TSC ticks per vector
(2^18 vectors), median of 15. Sources, scripts, raw numbers:
`lc-opt-explore/bitwise_xor_convolution/`.
- Phases of main (probe, 61 runs, ms): start 1.08, parse 3.69, rows 0.97 (both arrays), columns a
  0.60, columns b 0.86, inverse rows 0.78, format 0.66, `write()` 3.59, input `munmap` 0.74,
  exit 0.20. Kernel micro, hot (data touched first) against from L3: inverse rows 7.31 / 8.11,
  columns b 9.25 / 9.83; two radix-16 passes over a row 2.90, over a strip 2.54; reading and
  writing every strip line in column order 1.39.
- Zen 3 throughputs (`ports*.cpp`, 12 independent ops, core cycles per op): `vinserti128` from a
  register or from memory 1.0, `vperm2i128` 1.0 (register or memory), `vpermq` 1.27, `vcvtdq2pd`
  1.0 (and 1.44 per pair with an FMA), `vpmovzxdq` from memory, unpacks, `vshufps`, `vpsrlq` 0.5,
  `vpblendd`, `vpand`, `vpaddq` 0.25. A 128-bit store costs a full store slot (1 per cycle, as
  256-bit ones). 256-bit loads 2 per cycle; 4 adds + 1 store + 1 load per cycle sustain.
- Kept: the inverse's last pass does the 64-bit stage, a level, the 128-bit stage, a level, so
  the reduced values pack with `vshufps` only (no `vpermq`): kernel micro 8.11 → 7.33 (L3).
- Kept: products split in halves, the next row's h and M + l issued before this row's q and r:
  columns b 9.85 → 9.10, the product itself 4.7 → 3.6 core cycles per vector (latency bound
  before: ~25-cycle chain, ~5 rows in flight).
- Kept: the inverse's last pass a group of four ahead of its reductions: 7.29 → 6.99.
- Kept: `MADV_SEQUENTIAL` on the input (as `bitwise_and_convolution`): `munmap` 0.731 → 0.701.
- Probe, 61 runs: total 13.24 → 12.94 ms; columns b 0.860 → 0.801, inverse rows 0.775 → 0.644,
  `munmap` 0.741 → 0.701. `judge.py bench`, `lc-bench`, slowest 3 cases: 61 rounds 13.05 →
  12.93 ms (0.9828); an earlier 31 rounds 14.02 → 13.92 (0.9949, the VM was noisy).
- No gain: rows in 2 KiB blocks (first pass and vector bits 2-5 per block while in L1, then bits
  6-9): forward 5.45 → 5.63, inverse 7.32 → 7.70 through `transform`, 5.32 → 5.32 and 7.13 →
  7.31 with direct loops; products two rows ahead 9.13 → 9.29; the divide of a's columns a row
  ahead 5.09 → 5.67; a's values converted by `vpmovsxdq` and the magic constant instead of
  `vcvtdq2pd`: 8.35 → 8.94 (hot); prefetch hint T1 in the column passes 5.11 / 9.04 against
  5.11 / 9.05; output in whole 25600-value blocks across chunks (41 `write` calls, not 48),
  text in the first chunk's dead rows: wall ratio 1.000 (101 runs); `levels<4>` with the width
  as a template argument: two row passes 2.82 → 2.58 alone, but forward and inverse rows
  unchanged in the kernel micro.
- Radix-16 kernel in L1 (512 vectors, `xr*.cpp`): 21.4 cycles per call (1.34 per vector)
  against 17 for its 17 stores; register-only butterfly networks run at 4 adds per cycle. Also
  21.4: a software-pipelined loop (the next call's level-8 loads between this call's level-1
  stores; 9 register moves close the loop), stores 34 KiB away (out of place), each input
  loaded once (19 loads, not 26), loop aligned to 64 bytes. Loads replaced by register
  operands: 17.15 (store bound); keeping 10 or 18 of the 26 loads: 18.2. Cause not found.
- Not applicable: `convolution_mod`'s fixed-stride input path; max_random tokens have mixed
  lengths (20738704 bytes for 2^21 + 1 tokens).
- Checks: 13/13 official tests (`judge.py test`, `lc-amd`); `stress.py` 200 rounds and the three
  N = 20 known-answer cases (judge image); ASan/UBSan on all 13 official inputs, file and pipe;
  outputs of every variant byte-identical to main on max_random_00.
- Also no gain: the forward first pass in the inverse's stage order (64-bit stage, a level,
  128-bit stage, a level): kernel micro 5.42 → 5.40.
- PR #249 merged; CI 0.9896 (EPYC 7763 0.9849 and 0.9895, EPYC 9V45 0.9944). `judge.py bench`
  against main after the rebase onto `lib/run` (41 rounds): 12.91 → 12.79 ms (0.9921).
- Submitted the #249 `main.cpp` twice (2/5 this round):
  [409529](https://judge.yosupo.jp/submission/409529) AC 14 ms (max_random_00/01/02 12/14/13;
  large_02 13 against its usual 3, a spike), [409530](https://judge.yosupo.jp/submission/409530)
  AC 13 ms (13/13/13; small_00 9 ms, a spike). Best judged stays 13 ms: the ~0.15 ms gain is
  below the judge's 1 ms resolution and its run-to-run spread.

2026-10-10, claude, round 4 (`agent/bitwise_xor_convolution-r4`). Judge image and flags; timing
on `lc-k68` (EPYC 7B13, Linux 6.8 as the judge) unless noted. Probe and micro programs, scripts and
raw numbers: `lc-opt-explore/bitwise_xor_convolution/r4/`. Wall times fork to exit, medians of 31
runs unless noted.
- Phases of main (probe, 41 runs, ms): start 1.13, init 0.04, parse 3.99 (incl. 4 huge-page
  faults of x, ~0.35), rows 0.96, columns a 0.70, columns b 0.81, inverse rows 0.64, format 0.66,
  `write()` 3.81, input `munmap` 0.64, exit 0.21; total 13.67.
- Start: an empty program (`.preinit_array` entry, `_exit`) takes 1.31 ms dynamic, 0.54 static.
  The image's g++ links without `--as-needed`: libstdc++, libm and libgcc_s are always loaded.
  ld.so itself (`LD_DEBUG=statistics`) takes 222-508 K TSC ticks (0.07-0.17 ms); the rest is
  kernel work on the libraries' mappings. Nothing in the source can drop it.
- Input, kernel side (micro: separator scan of the 20.7 MB input): mapped 1.83 + `munmap` 0.60
  ms; `read()` into a reused page-aligned buffer at d = 0: 2.25 (128 KiB), 2.33 (64 KiB), 2.28 +
  0.48 for the buffer's huge page (1 MiB); at d = 16: 8.67. `MADV_POPULATE_READ` 1.50 against
  ~1.2 for the faults it saves. Whole parse (`read_bulk`, 2^16-token calls): mapped 3.60 + 0.70
  `munmap`; `read()` streaming 4.29-4.43 plus its buffer. No gain, as lib/io round 2 found.
- Parse alone on a populated mapping (ms per 2^21 tokens) by tokens per `read_bulk` call: 2^14
  2.82, 2^16 2.36, 2^18 2.18, 2^20 2.11. Each call ends with ~18 shrinking chunks and ~900
  tokens parsed one at a time. Round 2 lost with longer calls because rows then left the cache.
- Kept: `progress_read.hpp`, `BulkParser32` with a callback after each chunk, one call per array;
  the array's values at the end of x, each row widened and transformed as soon as it is parsed.
  Probe: parse + rows 4.95 → 4.75 ms; wall 13.76 → 13.41 (0.978, 41 runs). Chunks of 2^18 bytes
  instead of 2^17: 0.9907 and 0.9927 (41 and 51 runs); 2^16 1.0046, 2^19 1.0014.
- No gain: output chunks of 2^14 or 2^15 values instead of 2^16 (0.9988, 1.0007).
- Page faults per phase (`getrusage`): init 9, parse 344 (316 input, 8 of x, the parser's
  statics), columns a 2. No huge page is read before it is written.
- L1 bank conflicts as the radix-16 cause (round 3's open question): all 16 inputs of a call sit
  at the same offset in their lines when the step is even. Odd steps are not faster (core cycles
  per call: step 8 21.4, 9 22.8, 16 21.4, 17 22.2, 31 21.0, 32 21.4). Not the cause.
- `judge.py bench`, 41 rounds, slowest 3 cases: `lc-k68` 14.05 → 13.73 ms (0.9800), `lc-bench`
  13.97 → 13.60 (0.9796).
- Checks: 13/13 official tests on `lc-amd` and `lc-intel`; `stress.py` 300 rounds (every third
  input now with irregular whitespace, which takes the parser's slow path) and 4 known N = 20
  cases (one irregular) on both; ASan/UBSan on all official inputs, file and pipe; each parser
  path forced on the other CPU (`-DIO_BULK32_TRANSPOSE=0` on `lc-amd`, `=1` on `lc-intel`), with
  ASan/UBSan, on all official inputs and 4 irregular ones (N = 10, 12, 14, 20): outputs equal.

## Next

- Where 13.4 ms go (`lc-k68`): fixed kernel and loader work ~7.6 (start 1.13, input faults 1.2
  and `munmap` 0.65, huge-page zeroing 0.53, `write()` 3.8, exit 0.21); parse ~2.1; transforms,
  products and reductions ~3.0; format 0.66.
- lib/io: a progress callback in `io::read_bulk` would give every problem that consumes values
  chunk by chunk the gain above; then delete `progress_read.hpp`.
- Radix-16 passes are ~1.4 ms of it at 1.34-1.63 cycles per vector; 1.07 would be the store
  bound. The kernel study above found no lever yet.
- The 128-bit transpose stages (`vinserti128`, `vperm2i128`) take one cycle each on one pipe:
  4 per 16 values in the forward first pass and in the inverse last pass. Doing them with
  broadcast loads and blends needs a store and reload (an extra pass) or a layout where the
  swapped bit is transformed in another pass; sketched, not built (estimate 0.1-0.15 ms).
- Product fused into the column pass's radix-16 (multiply 16 inputs, then butterflies) would
  save a pass over each strip; estimate ~0.07 ms, needs a generated kernel.

## Sources

- Walsh-Hadamard transform for xor convolution: https://en.wikipedia.org/wiki/Hadamard_transform
- Montgomery reduction: P. L. Montgomery, Modular multiplication without trial division, Math.
  Comp. 44 (1985).
- int64/double conversion with the 1.5 * 2^52 constant: https://stackoverflow.com/questions/41144668
- Exact product as h + l with FMA (error-free transformation): standard; e.g. Ogita, Rump, Oishi,
  Accurate sum and dot product, SIAM J. Sci. Comput. 26 (2005).
- `.preinit_array` start: taken from `../convolution_mod/solution.cpp`.
- `MADV_SEQUENTIAL` on the input: taken from `../bitwise_and_convolution/solution.cpp`.
- Software pipelining (round 3): the standard technique, e.g. M. Lam, Software pipelining: an
  effective scheduling technique for VLIW machines, PLDI 1988.
- Zen 3 costs: measured here; see AGENTS.md for uops.info.
- `progress_read.hpp` (round 4): `BulkParser32` copied from `lib/io/bulk32.hpp` (ours), plus a
  callback.
