# lib/io

Input and output for every submission. API and usage: the header of `io.hpp`.
Tests: `lib/io/test.cpp`. In-memory timing: `lib/io/bench.cpp`.

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
| uint32 < 998244353 | 2.06 | 2.26 | 3.83 | 2.53 |
| uint32 0..9 | 1.59 | 2.11 | 3.69 | 2.47 |
| uint32, random bit length | 2.19 | 2.27 | 3.88 | 2.54 |
| uint64 <= 10^18 | 2.97 | | 5.28 | |
| uint64 < 10^16 | 2.95 | | 5.26 | |
| uint64 < 10^16, MaxDigits 16 | 2.27 | | 4.03 | |
| int64, random | 3.29 | | 6.04 | |
| int, random sign | 2.19 | | 4.60 | |

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
- Bulk reads for 64-bit and signed values.
