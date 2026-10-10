# lib/io

Input and output for every submission. API and usage: the header of `io.hpp`.
Tests: `lib/io/test.cpp`. In-memory timing: `lib/io/bench.cpp`. Whole-process floors of the
convolution problems: `problems/convolution/floor.py`.

## Design

- Input: a regular stdin file over 64 KiB is mapped between an anonymous zero page and at least 64
  zero bytes (the rest of the last page, or one more anonymous page). Smaller files and pipes are
  read into an aligned heap buffer with 64 zero bytes on each side: fewer system calls for tiny
  inputs. The guard before the input lets parsers load the 32 bytes that end at a token.
- Token boundaries: a 64-bit mask of whitespace per aligned 64-byte block. The next token clears one
  bit (`tzcnt`, `blsr`), so a read never waits for the previous token's parse.
- Values: the 16 (32 for 64-bit types) bytes that end at the token, masked to its length, then
  `pmaddubsw`, `pmaddwd`, `packusdw`, `pmaddwd`. A '-' saturates to 0, so signs cost one compare.
  `read<T, MaxDigits>()` with MaxDigits <= 16 parses 64-bit values in 16 bytes.
- Bulk `uint32_t` read: 128 KiB chunks cut into four streams at token boundaries, parsed in lockstep
  two tokens per step. Chunks shrink near the end of the array; the last < 1024 tokens use the
  scalar path. Irregular whitespace in a chunk falls back to one token at a time.
- Bulk `uint64_t` read (`bulk64.hpp`, `io::read_bulk(in, dst, n)`): the same chunks and streams.
  A step finds two tokens in the separator mask of the 64 bytes at a stream's position; each
  token's digits are the 32 bytes that end at it, minus a row of `kDigitMask` (0xFF before the
  token, '0' in it) with unsigned saturation. Values are 8-digit limbs joined in vectors:
  top * 10^16 + mid * 10^8 + low. It needs no input end: n tokens span at least 2n - 1 bytes,
  which bounds chunks and loads. It works through `Reader::scan()`/`resume()` and has its own
  header so that io.hpp, and the code of every problem bundled from it, stays unchanged.
- Bulk `uint32_t` read on Zen 3 (`bulk32.hpp`, `io::read_bulk(in, dst, n)`): the same streams and
  steps, but each step stores its eight values as one vector, [a0 a1 a2 a3 b0 b1 b2 b3] for the
  two tokens of each stream, into a buffer; a 4x4 transpose of value pairs then writes each
  stream's values in order. Elsewhere `io::read_bulk` is `Reader::read`; `-DIO_BULK32_TRANSPOSE`
  forces either. Like `bulk64.hpp`, it needs no input end.
- Output: 64 KiB buffer, `write(2)`. Integers: 4-digit table (10000 entries), groups placed in a
  vector, `pshufb` drops the leading zeros, one 16-byte store. Digit count from a 32-entry table
  (32-bit) or two 65-entry tables (64-bit). No branches on value size. `write<MaxDigits>()` with
  MaxDigits <= 16 skips one 64-bit division. `write_with()` hands the buffer to custom formatters.
- Bulk `uint32_t` write: eight values per step in AVX2. Digit counts come first, from vector
  compares, so the store addresses do not wait for the digits.

## Measurements

AMD EPYC 7B13 (`lc-amd`, the judge's CPU), GCC 15.2 in the pinned image, judge flags, 2026-10-09.

In memory, ns per token, median of 15 rounds of 2^20 tokens (`bench.cpp`):

| Input | read | bulk read | write | bulk write |
|---|---|---|---|---|
| uint32 < 998244353 | 2.07 | 2.27 | 3.80 | 2.53 |
| uint32 0..9 | 1.59 | 2.11 | 3.69 | 2.47 |
| uint32, random bit length | 2.18 | 2.27 | 3.81 | 2.53 |
| uint64, full range | 2.98 | 3.91 | 5.31 | |
| uint64 <= 10^18 | 2.97 | 3.90 | 5.33 | |
| uint64 < 10^16 | 2.95 | 3.88 | 5.30 | |
| uint64 < 10^16, MaxDigits 16 | 2.26 | 3.87 | 4.05 | |
| int64, random | 3.28 | | 5.95 | |
| int, random sign | 2.18 | | 4.60 | |

Whole process, judge-like runner, ms, median of 21 rounds, max over the 3 largest cases:

| Program | ms |
|---|---|
| empty `main` | 1.13 |
| convolution_mod floor, read only: old QPoly I/O (submission 408716 minus the NTT) | 3.80 |
| same, lib/io (destination in 2 MiB pages, as the old one) | 3.52 |
| convolution_mod floor, read and print: old QPoly I/O (fixed-width output) | 10.16 |
| same, lib/io (`write_array`, variable width) | 11.18 |
| many_aplusb: `scanf`/`printf` | 270.60 |
| many_aplusb: lib/io, one value at a time | 31.90 |
| many_aplusb: decimal addition in vectors (its `main.cpp`; record 23 ms) | 21.12 |
| aplusb: `scanf`/`printf` vs lib/io, ratio | 0.99 |

Floors (`problems/convolution/floor.py`, 11 rounds): read the slowest test's input, write an
answer of the same length, nothing else. Arrays in huge pages; "fixed" prints 9-character fields
(`problems/convolution/convolution_mod/fields.hpp`, judge-specific) where every value is below
10^9; that column is from round 5, the others from round 1 (round 1's fixed column used the
slower `fixed_width.hpp`: 9.61, 230.1, 11.88, 11.76, 11.87, 11.19, 11.57, 6.46, 3.89, 3.88).
Record: the fastest judged time when the problem's issue was opened. 2_64 rows: 21 rounds.

| Problem | in / out MB | floor ms | fixed ms | record ms | best floor / record |
|---|---|---|---|---|---|
| convolution_mod | 10.5 / 10.4 | 11.00 | 9.04 | 14 | 65% |
| convolution_mod_1000000007 | 10.5 / 10.4 | 10.97 | | 29 | 38% |
| convolution_mod_large | 335.5 / 331.8 | 284.0 | 217.4 | 452 | 48% |
| convolution_mod_2_64 | 22.0 / 21.4 | 18.67 | | 76 | 25% |
| convolution_F_2_64 | 22.0 / 21.4 | 18.94 | | 409 | 5% |
| min_plus_convolution_convex_convex | 10.5 / 11.5 | 11.31 | | 20 | 57% |
| min_plus_convolution_convex_arbitrary | 10.5 / 11.0 | 11.36 | | 38 | 30% |
| min_plus_convolution_concave_arbitrary | 10.4 / 11.0 | 11.58 | | 117 | 10% |
| bitwise_and_convolution | 20.7 / 10.4 | 13.53 | 11.26 | 26 | 43% |
| bitwise_xor_convolution | 20.7 / 10.4 | 13.39 | 11.14 | 25 | 45% |
| mul_mod2n_convolution | 20.7 / 10.4 | 13.15 | 11.26 | 81 | 14% |
| gcd_convolution | 19.8 / 9.9 | 12.71 | 10.96 | 37 | 30% |
| lcm_convolution | 19.8 / 9.9 | 12.66 | 10.79 | 37 | 29% |
| mul_modp_convolution | 10.4 / 5.2 | 7.29 | 6.18 | 45 | 14% |
| multivariate_convolution | 5.2 / 2.6 | 4.19 | 3.71 | 117 | 3% |
| multivariate_convolution_cyclic | 5.2 / 2.6 | 4.18 | 3.69 | 117 | 3% |

The 2_64 rows use the bulk uint64 read (main: 19.87 and 20.52). The uint32 floors parse with
`Reader::read`; the solutions use `io::read_bulk` (`bulk32.hpp`), 0.1-0.4 ms less (round 3).

The 64-bit bulk read loses in `bench.cpp` (21 MB of text streamed from memory) but wins on warm
input (`BulkParser64::parse` 1.6 ns per token for 2^14..2^20 tokens, a scalar chain of separator
and `parse_ending24` 5.0) and in the whole process (below).

The old code prints every value in 10 columns (the checker allows it), about 1.1 ns per value.
That is a judge-specific trick: it belongs in a problem's `main.cpp`, not here.

many_aplusb, where the time goes (ms): start 1.1, input pages 4.7, parse 2M tokens 5.9, format
1M values 10.4, `write()` of 20 MB 9.4.

## Log

2026-10-09, claude (all on `lc-amd` unless noted):
- Pointer-chasing scalar read (find separator, then advance): 4.8 ns/token (uint32), 9.3 (10^18).
  Separator-mask reads: 2.1 and 3.7. Kept.
- Bulk read with token masks, four tokens per step: 1.86 ns/token; four-stream lockstep: 1.56-2.6
  depending on cache state. End to end within 1.3% of each other. Kept the four-stream parser.
- Four-stream parser with its stream pointers behind a reference: 0.3 ms slower than the old
  parser on 1M tokens. Local copies: 0.15 ms slower. Chunks shrinking at the end instead of a
  scalar tail of 65K tokens: 0.28 ms faster than the old parser.
- First version end to end: many_aplusb 48.9 ms, convolution_mod floor 17.7 ms (formatting 7.5).
- Formatters, uint32 < 998244353, ns per value in a plain loop: SWAR 8 digits 5.9; SWAR with a
  digit-pair head 4.7; 4-digit table with branches 2.7 (3.7 on mixed sizes); 4-digit table and
  `pshufb`, no branches, 2.7 on every size. Kept the last.
- uint64 formatter: three branches by size 5.6 (10^18) but 8.5 on random sizes; no branches 5.8
  on both. Kept no branches.
- Bulk write: digit counts from the finished digits made each store wait: 3.45 ns. Counts from
  vector compares first: 2.42. Shuffle controls built in vectors: 2.74. Kept 2.42. End to end
  6.5% faster than one `write` per value.
- Input pages, 40 MB: lazy `mmap` 4.6 ms; `MAP_POPULATE` +0.8; `MADV_POPULATE_READ` +1.8;
  `read()` into a 2 MiB-page buffer +2.2; `read()` into `malloc` +28.
- `write()` of 10 MB into the output file: 4.9 ms, 20 MB: 9.4 ms. Not reducible from user code
  (see QPoly exploration 011: `fallocate`, mapped output, larger writes all lose).
- aplusb (tiny input) on CI's EPYC 7763: mapped input 1.2% slower than `scanf`; `fstat`, `lseek`,
  `mmap`, `munmap` and a page fault against one `read()`. Files up to 64 KiB are now read.
- Writer state kept out of escaping calls (so GCC can keep it in registers): no change measured.
- Assumptions, ns per token, to decide which deserve options:
  - uint64 10^18: 32-byte load ending at the token 3.01 vs two 16-byte lanes 3.92. Default now.
  - uint64 < 10^16: one 16-byte load 2.55 vs 3.73. Option `MaxDigits <= 16`.
  - Signed, non-negative: `read<long long>` 4.44 vs `read<uint64_t>` 3.98; `read<int>` 2.61 vs
    `read<uint32_t>` 2.14. With the end-anchored load the sign costs one compare: int64 3.29.
  - Exactly one separator between tokens (skip the empty-token check): no gain. No option.
  - uint64 format, < 10^16 assumed: 2.95 vs 5.55; full range via x / 10^8 twice: 4.65 vs 5.57
    (one division by 10^16). Both kept.
- Streamed input, a 256 KiB buffer refilled with `read()`, as a Reader template option: reading
  alone 1.1 ms faster than mapping on 40 MB, but many_aplusb 24.6 vs 21.7 ms (user +1.9 ms, system
  +1.5 ms with the input on tmpfs). Removed.
- Software prefetch 512 B-8 KiB ahead of the separator scan: +0.8% to +3.6% on many_aplusb.
- `write()` chunk size, 20 MB: 16 KiB +9%, 64 and 256 KiB equal, 1 MiB +1%, one 20 MB write +94%.
- Input mapping, 40 MB: `MAP_SHARED` equal to `MAP_PRIVATE`; `MAP_SHARED | MAP_POPULATE` +15%.
- uint64 batch write (scalar split into three 8-digit limbs, AVX2 digits for eight values, three
  stores per value): 5.5 ns per value vs 4.4 for the scalar formatter in the same loop. Not kept;
  the limb split and digit count are scalar either way.
- Exit: `std::_Exit(0)` after `flush()` instead of returning from `main` skips the exit handlers,
  0.03 ms per run (aplusb 1.15 → 1.12 ms). Solutions do this; the library cannot.
- `Reader::scan()`/`resume()`: the separator state for custom token loops (many_aplusb's assembly).
- Writer buffer as a separate allocation (stores through it cannot alias the Writer, so GCC keeps
  the cursor in a register): in memory, one `write()` per value 3-8% faster. Whole process, 1M ints
  or int64 written one by one: 1.005 and 1.007. Not kept.
- `write_array` for other integers checking the buffer once per 1024 values: in memory 10-30%
  faster; whole process (1M ints and 1M int64): 1.006. Not kept. In-memory gains in formatting
  do not reach the whole process here; check end to end before keeping a Writer change.
- Bulk parser, from its compiled loop: the four output pointers were reloaded every step (an array
  passed by reference; vector stores may alias it) and 32-bit lengths were widened for each table
  index. Local copies and 64-bit lengths: convolution_mod read floor 0.977. A sliding-window
  right-align table (row = the 16 bytes at length + 1) instead of 16-byte rows saves the multiply
  by 16: 0.968 in total; in memory 2.39 → 2.26 ns per token. The loop is now ~11 instructions per
  token; assembly would save little.
- Judge harness (`tools/judge.py`): inputs now on tmpfs, as on the judge, instead of the VM's disk
  cache. Same timings for these programs.
- Writer buffer for large outputs, convolution_mod_large (331 MB out), output phase (ms, `lc-amd`):
  64 KiB 155.6, 160 KiB 151.9, 256 KiB 151.1, 1 MiB 151.3. As a template capacity
  (`BasicWriter<Capacity>`, `Writer` the 64 KiB alias): the sample problems' code is unchanged,
  but aplusb's functions moved; CI measured aplusb at 1.006 and 1.014 (Intel 8573C 1.027, 1.033)
  and blocked it. Not kept. Instead the convolution formatter fills 250 KB blocks and passes each
  to `write(std::string_view)`, which sends strings longer than the buffer straight to `write(2)`:
  output 144.3 ms. Large outputs need no Writer change.

2026-10-09, claude, issue #21 round 1 (`lc-amd`, judge flags; floors with `floor.py`, ratio to main):
- I/O floor harness (`problems/convolution/floor.cpp`, `floor.py`): reads a problem's input and
  writes an answer of the same length and value range; times it on the 3 largest tests, several
  io.hpp versions in one run. Table under Floors. Its first version kept a and b in `new[]` arrays:
  floors 1.9 to 63 ms higher (convolution_mod 12.93, bitwise_and 16.95, convolution_mod_large 347
  ms), page faults on 4 KiB pages. Now one 2 MiB-aligned `MADV_HUGEPAGE` mapping, as the solutions do.
- Bulk `uint64_t` read (`BulkParser64`) as a branch of `Reader::read`, floor ratios to main (21
  rounds): convolution_mod_2_64 0.941 (19.87 → 18.67 ms), convolution_F_2_64 0.913 (20.52 →
  18.94); a control without it 1.002 and 1.011. With the first harness: 0.927 and 0.930. All 95
  official inputs of both problems parse the same as one token at a time.
- In io.hpp (PR #46) it changed every bundled `main.cpp`. The six problems on lib/io compiled to
  byte-identical `.text` and `.rodata` (judge flags, `-march=znver3`), yet CI's strict gate called
  five of them slower: aplusb 1.0104 (Xeon 6973P-C 1.026), gcd_convolution 1.0062, many_aplusb
  1.0046, convolution_mod_large 1.0006, bitwise_and_convolution 1.0001. If each passes on noise half
  the time (a guess), all six pass 1 time in 64. Moved to `lib/io/bulk64.hpp` (`io::read_bulk`),
  without the input end: io.hpp is unchanged and no problem is re-timed. Floors in one run,
  `io::read_bulk` vs `Reader::read` (21 rounds): convolution_mod_2_64 0.904, convolution_F_2_64
  0.917. Kept.
- Bulk `uint64_t` write (pass 1 scalar: three 8-digit limbs and the digit count; pass 2 AVX2:
  20 digits for eight values, one 32-byte store per value ending at its last digit, separators
  blended in, stores right to left so each overwrites the previous one's leading bytes). In memory
  5.16 vs 5.31 ns per value; floor of convolution_mod_2_64 with it 0.953, without it 0.903, scalar
  read with it 1.040. Removed: `write_array` writes uint64 one value at a time.
- Chunk size of the 64-bit parser, floor ratios on 2_64 and F_2_64: 2^17 0.927 and 0.930, 2^16
  0.934 and 0.951, 2^15 0.928 and 0.938, 2^14 0.952 and 0.949, 2^13 0.990 and 0.999. Kept 2^17.
- Chunk size of the 32-bit parser (convolution_mod, bitwise_xor_convolution, min_plus convex_convex;
  41 rounds, control: a copy of main): 2^15 1.013, 1.014, 0.999; 2^16 1.011, 1.007, 1.008;
  control 1.005, 1.009, 0.992. A first run of 21 rounds had shown 0.95 for 2^15: noise. Kept 2^17.

2026-10-09, claude, issue #21 round 2 (`lc-amd`, judge flags, `floor.py --fixed`, ratio to `Reader`):
- New reason to retry streamed input: bitwise_and_convolution measured `read()` into a reused
  64 KiB buffer at 1.87 ms against 2.16 ms for mapping, touching and unmapping its 20.7 MB input,
  and the convolution problems read all input before any output. `io::StreamReader` in a new
  header: 64 KiB refills, `BulkParser` per refill, limit at the start of the last complete token
  (its scans past the chunk then stay in the buffer; a limit at the last separator let a long
  whitespace run read past the buffer, caught by the test). Passed the lib tests with judge flags
  and ASan/UBSan, file and pipe, 4 KiB and 64 KiB buffers.
- Floors, 21 rounds (convolution_mod_large 7): bitwise_and 11.82 → 12.41 ms (1.039),
  convolution_mod 9.44 → 9.69 (1.018), gcd 11.28 → 11.63 (1.026), convolution_mod_large
  222.3 → 229.4 (1.033). Removed. Why (a guess from the phases): with the mapping, the DRAM
  reads overlap the parse and only fault-around and `munmap` (1.64 ms per 20.7 MB) are extra;
  the kernel copy of `read()` runs serially and costs more than that.

2026-10-10, claude, issue #21 round 3. Parse alone: warm input of bitwise_and max_random_00 (2^21
tokens, 90% of 9 digits), in-process, ns per token, medians of 21; "in L2" parses 16K tokens
301 times. `lc-amd` at 3.48 GHz unless noted.
- `Reader::read` (BulkParser): 1.17 (in L2 1.29), ~11 instructions per token, IPC ~2.7.
- Block parser: aligned 64-byte blocks, the tokens ending in a block in 8 lanes (positions by
  `tzcnt`/`blsr`, windows ending at each separator, rows by length), no pointer chain: 1.23
  (in L2 1.20). 18.75 instructions per token at IPC 4.4: bound by instruction count, a quarter
  of the lanes empty. `lc-intel`: 0.99 vs 1.06.
- More streams for BulkParser (pointer chains): 8 1.29, 12 1.31. Four tokens per step (64-byte
  mask, windows ending at the separators): 1.20. Masks from a separator bitmap built per chunk
  (the chain without `vpmovmskb`): 1.33.
- Ablations of BulkParser (values wrong): no stores 0.99; one 32-byte store per step instead of
  four 8-byte stores, `vpermd` and `vextracti128` 0.97; constant rows 1.15; no window insert 1.07;
  constant stride (no pointer chain, rows hoisted) 0.75.
- Kept: one store per step and a transpose pass (`BulkParser32`, `bulk32.hpp`): 0.97 (-17%), in L2
  1.15 (-11%); 0.94 vs 1.13 on convolution_mod all_same_01. `lc-intel` (Emerald Rapids,
  x86-64-v3): 1.07 vs 1.02 (+5%), so `io::read_bulk` takes it only on `__znver3__`.
- Transposing in registers every four steps instead (four 32-byte stores to the streams): 1.09 on
  `lc-amd`, 1.04 on `lc-intel`. Row pairs from one 32-byte load of a 35 KB table: -1%. 8 streams
  with whole-vector stores: slower than 4. None kept.
- Floors with the new parser in `Reader::read` (`floor.py --fixed`, 21 rounds): bitwise_and
  11.74 → 11.46 ms (0.972), gcd 11.36 → 10.97 (0.966), convolution_mod 9.61 → 9.50 (0.982),
  mul_modp 6.61 → 6.41 (0.971).
- Problems switched to `io::read_bulk` (`judge.py bench`, 21 rounds, slowest 3 cases, ms, ratio):
  bitwise_and 13.17 → 12.79 (0.979), gcd 15.13 → 14.86 (0.979), bitwise_xor 14.97 → 14.68
  (0.981), lcm 16.37 → 16.05 (0.981), min_plus convex_arbitrary 11.25 → 11.05 (0.981), mul_mod2n
  20.57 → 20.24 (0.986), convolution_mod_large 412.06 → 406.55 (0.987),
  convolution_mod_1000000007 24.03 → 23.86 (0.989), min_plus concave_arbitrary 28.20 → 28.03
  (0.991), multivariate_convolution_cyclic 11.36 → 11.25 (0.991), mul_modp 12.04 → 11.93 (0.993).
  multivariate_convolution 13.71 → 13.68 (0.998, 2^19 tokens): noise, left on `Reader::read`.
  Not switched: convolution_mod (another round is running on it), min_plus convex_convex (own
  parser), the polynomial problems (another session).
- Checks: lib tests with judge flags, with `-DIO_BULK32_TRANSPOSE=0`, and with ASan/UBSan, on
  `lc-amd` and `lc-intel`; the transposed parser is tested on every CPU through
  `detail::read_transposed`. Every switched problem passes all official tests; ASan/UBSan builds
  pass on the 3 largest cases of each (but convolution_mod_large), file and pipe input.
- PR #176 merged; CI geomean 0.9885 over the 11 problems. EPYC 7763 (Zen 3) runs 0.961-0.993
  (one bitwise_and run 1.010); EPYC 9V74/9V45 and Xeon 8573C run the unchanged path: 0.98-1.014.
- Submitted 5 (the round's cap): gcd 409380 14 ms (was 15); bitwise_xor 409379 14, lcm 409381
  16 (unchanged); bitwise_and 409382 21 and min_plus convex_arbitrary 409383 15, launch spikes
  (`tools/spikes.py`: clean 12 and 11, unchanged). Details in each problem's notes.

2026-10-10, claude, issue #21 round 4: the kernel side of the floors. `lc-amd`, GCC 15.2 image,
in Docker with a 1 GiB memory limit and tmpfs files unless noted; medians of 21-41 rounds.
- Where a floor goes (`perf record -a` on `lc-intel`, bitwise_and floor, 1500 runs): 45% user,
  38% kernel, 3.3% `ld.so`. Kernel: output page cache ~12% (`shmem_add_to_page_cache` 4.4, copy
  1.9, page allocation, LRU and memcg the rest), huge-page zeroing 5.0 (`kernel_init_pages`), input
  fault-around ~5, input unmap ~3.
- `write(2)` of 10.5 MB to a new tmpfs file: 64 KiB chunks 3.07 ms, 256000 B 3.00, 256 KiB 3.01,
  1 MiB 2.99; `ftruncate` first 3.06; `fallocate` first 3.38; `writev` of 4 pieces 3.02. Same on
  `lc-intel`: 1.16 ms (9 GB/s; the AMD VM's kernel paths are ~3x slower, a guess: the Zen 3
  mitigations listed in its sysfs, "Safe RET" and "Clear CPU buffers").
- The file offset against the buffer, d = (offset - buffer) mod 4096: d = 0 3.19 ms; d = 1..16
  6.50-6.66 (2.06x); d = 32..192 3.31-3.43; d >= 256 3.22-3.27. Cause: the kernel copies with
  `rep movsb`, which on Zen 3 runs at 3.0 GB/s for d = 1..31 against 47 GB/s for d = 0 (user-space
  test, 4 KiB copies; d = 48: 33). `lc-intel`: 24-26 GB/s at every d, `write(2)` unaffected.
  glibc 2.41 `memcpy`: 46-47 GB/s at every d. So: never let a long write start 1-31 bytes past
  its buffer, modulo 4096; whole pages from a page-aligned buffer are fastest.
- Scan of every solved problem's writes (an `LD_PRELOAD` shim logging d per `write(2)`, largest
  case, 5 runs): none in the slow band but convolution_F_2_64 (0.8% of its bytes).
- Input mapping, 20.7 MB (bitwise_and max_random_00), in-process: map 0.01, touch 1.29-1.32,
  `munmap` 0.62-0.66 ms. `MADV_RANDOM` or `MADV_SEQUENTIAL`: `munmap` 0.62 vs 0.66 (the kernel
  then skips marking each page accessed); `MADV_WILLNEED`: no change; `MAP_POPULATE`,
  `MADV_POPULATE_READ`: map 1.42, total +0.45. Dropping consumed input during the touch
  (`MADV_DONTNEED` every 256 KiB to 4 MiB, or `munmap` every 1 MiB): total 2.80-2.91 vs
  2.72-2.80; the unmap cost moves, it does not shrink. With a fresh file per round, as on the
  judge: same order. None kept: the -0.04 ms is below noise, and io.hpp would re-bundle every
  problem, those of running rounds included.
- `BulkParser64` with whole-vector stores and a transpose, as `bulk32.hpp`: warm parse of 2_64
  inputs 1.70 → 1.60 ns per token (-6%). That is 0.1 ms per 1M tokens, 0.25% of
  convolution_mod_2_64: below whole-process noise. Not kept.
- Kept, problem side: `problems/convolution/fixed_width.hpp` takes the text buffer from the
  caller, and `text_buffer()` places it page-aligned in dead, already-touched memory (else a
  static). The solutions pass b after the product (bitwise_and), the rest of a (bitwise_xor), g
  (multivariate_convolution), instead of a static 250 KB (63 page faults; bitwise_and 497 → 428
  faults per run on `lc-intel`). Blocks are 24576 values, 60 pages, so each write is whole pages
  at d = 0. `judge.py bench`, 31 rounds, slowest 3 cases: bitwise_and 12.93 → 12.66 ms (0.985;
  the buffer move alone 0.988), multivariate_convolution 13.76 → 13.59 (0.988), bitwise_xor
  14.70 → 14.61 (0.998; 41 rounds with a control: 0.991, control 0.995). Outputs byte-identical
  to main on every official test (judge build, ASan/UBSan, pipe input). `floor.cpp` places its
  text the same way.
- PR #187 merged; CI geomean 0.9871: bitwise_and 0.9882, bitwise_xor 0.9810,
  multivariate_convolution 0.9920 (EPYC 7763 0.980-0.987). Submitted 2: bitwise_and 409404
  and multivariate_convolution 409405, both with launch spikes; clean 12 and 14 ms, unchanged
  (the gain is 0.1-0.2 ms, under the judge's 1 ms resolution). Details in their notes.

2026-10-10, claude, issue #21 round 5: output. `lc-amd`, GCC 15.2 image, judge flags, in Docker,
files on tmpfs.
- `write(2)` alignment, re-measured: 10 MiB to a new tmpfs file in chunks from a page-aligned
  buffer plus a skew, medians of 31 rounds (a second run: 41). Whole-page chunks: d = 0 3.07 ms
  (3.16); d = 32, 64, 128, 192, 256, 512, 2048, 3584, 3840, 3968, 4032, 4064 3.07-3.08
  (3.13-3.16; d = 48 3.22, d = 4048-4095 3.17-3.19); d = 1, 16, 24 6.32-6.54 (2.06x).
  256000-byte chunks (d alternates 0 and 2048) 3.07 and 281600-byte chunks 3.07, as whole pages.
  64 KiB-sized chunks: 61440 bytes 3.14, 65536 3.13, 65516 (d drifts) 3.22. So only d = 1..31
  costs; round 4's 3.31-3.43 ms for d = 32..192 did not repeat, and whole pages gain nothing.
- The writes of every convolution problem and many_aplusb on its largest case (an `LD_PRELOAD`
  shim logging size and d per `write(2)`): none at d = 1..31 except convolution_F_2_64 (0.6% of
  its bytes). Two round-4 leads are void: page-multiple blocks for the other fixed-width writers,
  and Writer flushes of whole pages. Neither these nor `MADV_RANDOM` (-0.04 ms per 20 MB) is worth
  an io.hpp change that re-bundles every problem.
- Formatters: bitwise_and, bitwise_xor and multivariate_convolution printed with
  `problems/convolution/fixed_width.hpp` (1.04 ms per 2^20 values in memory), the other uint32
  problems with `convolution_mod/fields.hpp` (0.64 ms; `convolution_mod/notes.md`, round 2), the
  same text. The three now use fields.hpp, the text still page-aligned in dead memory
  (`problems/convolution/text_buffer.hpp`, fixed_width.hpp's `text_buffer()` made generic);
  fixed_width.hpp is deleted and `floor.cpp` uses fields.hpp. `judge.py bench`, 31 rounds,
  slowest 3 cases: bitwise_and 12.56 → 12.27 ms (0.973), bitwise_xor 14.53 → 14.14 (0.973),
  multivariate_convolution 13.69 → 13.60 (0.995; a quarter of the output). Outputs
  byte-identical to main on all official tests and 200 random inputs each, ASan/UBSan included.
- 64-bit: convolution_F_2_64's own fixed-width formatter takes 3.19 ms per 2^20 random values in
  memory, `convolution_mod_2_64/fields64.hpp` 2.62. F_2_64's long blocks now go through fields64,
  12288 values per direct `write(2)` from its dead b, instead of 65516-byte Writer flushes.
  gen_max_00 37.85 → 36.78 ms, many_ones_00 38.09 → 37.12; `judge.py bench`, 31 rounds, 6
  slowest cases (the variable-width ones unchanged): 39.62 → 39.23 (0.976). Details in its notes.
- Floors with fields.hpp (`floor.py --fixed`, 11 rounds): the fixed column above, 0.2-0.8 ms
  below round 1's (convolution_mod_large 12.7).

## Sources

- Our own QPoly explorations 007 and 011 (`../SymPoly/work/ntt/io_yosupo`, `io_large`): the
  four-stream parser and the vector digit weighting, rewritten here. The old QPoly input mapping
  and table writer came from another user's submission and are not used.
- Digit weighting with `pmaddubsw`/`pmaddwd`: https://kholdstare.github.io/technical/2020/05/26/faster-integer-parsing.html
- Separator bitmasks walked with `tzcnt`/`blsr`: simdjson, https://arxiv.org/abs/1902.08318
- Digit count tables: https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
  and https://lemire.me/blog/2025/01/07/counting-the-digits-of-64-bit-integers/
- 4-digit table formatters measured fastest without AVX-512: https://ibireme.github.io/c_numconv_benchmark/

## Next

- Bulk write is 1.3 ns per value slower than fixed-width output; the vector work, not the stores,
  is the limit (in-memory: compute 1.4, with movemask 1.7, full 2.4 ns per value).
- Bulk reads for signed values.
- Fold `io::read_bulk` (both widths) into `Reader::read` when io.hpp changes anyway, with gains
  that outweigh the noise of re-timing every problem; then switch convolution_mod and the
  polynomial problems too.
- `BulkParser64` with whole-vector stores: -6% of the parse, measured in round 4; worth adding
  only with another gain for the 2_64 problems, since alone it is below their noise.
- Writer: flushes of arbitrary size land at d = 1..31 with odds 31 in 4096, each then 2x slower
  (~0.8% of `write(2)` time on average); whole pages gain nothing more (round 5). Fold a fix in
  with the next io.hpp change, together with `MADV_RANDOM` on the input (-0.04 ms per 20 MB).
- `convolution_mod/fields.hpp`'s first comment still names `../fixed_width.hpp`, deleted in
  round 5 (that folder belonged to another round); fix it with the next change there.
- convolution_F_2_64's variable-width blocks (all_ones, all_same, small_values: 35-36 ms, near
  its slowest case) print one `uint64_t` at a time through `write_array`.
- The parser is still about 0.2 ns per token above the constant-stride ablation (0.75): the
  pointer chains. Tried and lost: more streams, four tokens per step, a bitmap of separators.
- A faster uint64 write: only if a problem's floor becomes a large share of its time
  (convolution_mod_2_64 and convolution_F_2_64 are at 25% and 5% now).
- Huge-page arrays (2 MiB-aligned mapping, `MADV_HUGEPAGE`) are copied in three solutions and cut
  floors by 15-20%: a shared helper may belong in `lib/`. It gains nothing on solved problems
  (all allocate this way already); add it with the next problem that needs it.
- Streamed input lost to the mapping on all four solved convolution problems (round 2); the rest
  of the floors is kernel time (`write()`, input faults, `munmap`, huge-page zeroing), the parser
  (~1.0 ns per token on Zen 3 with `bulk32.hpp`) and the formatter. Round 4 found no cheaper
  syscall pattern for any of the kernel parts but the write alignment above.
