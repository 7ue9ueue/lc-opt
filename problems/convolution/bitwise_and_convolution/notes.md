# bitwise_and_convolution

N <= 20; 2^N values a_i, b_i < 998244353; print c_k = sum over i & j = k of a_i b_j, mod 998244353.
5 s. Inputs are ~20.7 MB, outputs ~10.5 MB (fixed width) at N = 20.

Best judged: ours, [409492](https://judge.yosupo.jp/submission/409492), 11 ms (first; round 4,
all three max_random cases 11 ms). Before: 409202 and 409430, 12 ms; round 1: 409186, 13 ms.
Next other user: 26 ms (adamant, 400554).

## Design

- c = mu(zeta(a) * zeta(b)): zeta sums over supersets (x[i] += x[i | bit] for each bit), mu
  inverts it (x[i] -= x[i | bit]). 3 transforms of 20 levels and 2^20 products at the maximum.
- Arithmetic: values in [0, P). One butterfly is add, subtract P, unsigned min (3 vector ops); the
  inverse is subtract, add P, min. Product: Barrett, q = floor(floor(x y / 2^29) floor(2^61 / P)
  / 2^32) is floor(x y / P) or one less; x y - q P from low words (4 `vpmuludq`, 2 `vpmulld`).
- Layout: rows of 2^17 values (8 rows at N = 20). Bands of 4 rows are contiguous and read with one
  `Reader::read` call; each band starts 256 words (16 lines) after the previous one ends, so the 16
  rows of a and b spread over 4 groups of L1 sets. Each row is transformed over its own 17 bits
  after its band is parsed; one column pass then does the 3 row bits of a and b, the product and
  the inverse row bits; each row then gets its inverse 17 bits. Output with
  `../convolution_mod/fields.hpp`, its text in b (dead by then): per band, whole 25600-value blocks
  as rows finish and the rest at the band's end, so no write goes through the Writer's buffer.
- Inside a row: pieces of 2^12 values (16 KiB, L1), then the row's upper bits. The lane bits use a
  transpose: per piece, a radix-8 sweep over the 3 lowest vector bits, then per 8-vector tile an
  8x8 transpose and radix-8 over the former lane bits. The transpose's 128-bit stage is done by the
  loads (128-bit broadcasts and blends; shuffles run on 2 of Zen 3's 4 vector pipes). Forward tiles
  stay transposed (the product does not care); the inverse restores them. The forward pass
  prefetches the next piece (fresh from the parser, in L3). Sweeps do not store the last vector of
  a group (all bits set: no level changes it).
- The input mapping is advised `MADV_SEQUENTIAL` (`io::advise_sequential`, first here): its
  `munmap` then skips marking pages accessed.
- Shorter inputs are padded with zeros to 2^6 values; zeros do not change c_k for k < 2^N.
- Memory: a and b in one mapping, 4 huge pages (`MADV_HUGEPAGE`) and one 4 KiB page below them
  for the 4 KiB beyond 8 MiB.
- Runs from `.preinit_array` and ends with `_exit` (`RUN_EARLY`, lib/run): libstdc++'s
  initializers and exit handlers never run.

## Measurements

`lc-amd` (EPYC 7B13), GCC 15.2 in the judge image, judge flags, `tools/judge.py bench`, slowest 3
cases (all max_random, N = 20), 21 rounds, 2026-10-09, branch `agent/bitwise_and_convolution`.

| Program | Median ms | Ratio |
|---|---|---|
| v1: rows of 2^15, lane bits by in-register shifts | 14.60 | 1 |
| v2: rows of 2^17, transposed tiles, L1 pieces | 13.74 | 0.942 |
| floor: same allocation, reads and output, no transforms | 11.91 | 0.874 of v2 |

Round 2, same setup, 31 rounds: v2 13.64, v3 (this `main.cpp`) 13.16, ratio 0.968.

Phases of v3 (ms, max_random_01, medians of 61 runs, `CLOCK_MONOTONIC` stamps in a scratch probe;
wall time from fork to exit): start 0.94, parse 3.68, row transforms 0.73, column pass 0.38,
inverse row transforms 0.39, format 1.13, `write()` 3.42, input `munmap` 0.68, exit 0.20; total
11.61. Whole process, 61 interleaved runs: v2 12.13, v3 11.58 (ratio 0.955). The parse phase
splits into input fault-around 0.96, a/b huge page faults 0.23, parsing 2.47 (measured by
touching the pages first).

Round 1, v2: `perf` on `lc-intel` (static build): 40% of cycles in the kernel
(`shmem_add_to_page_cache` 11%, `kernel_init_pages` 9.5%, fault-around 6%); user: bulk parser
23%, transforms 18%, formatter 9%, `memmove` (parser streams, Writer) 3%.

## Log

- 2026-10-09, claude, round 1: first solution (v1, then v2 above). 13/13 official tests; stress
  test against `brute.cpp` (`stress.py`, 90 rounds, N <= 14, and N <= 9 with -DBLOCK_LOG=6 for
  1-8 rows); ASan/UBSan builds of both on every official test. Measured on `lc-amd`:
  - Row length (phases in ms: parse, row transforms, column pass, inverse rows): 2^15 rows 4.1,
    0.83, 0.93, 0.42; 2^16 3.97, 0.92, 0.72, 0.47; 2^17 3.95, 0.96, 0.55, 0.50. Fewer rows: fewer
    `read` calls (each ends with shrinking chunks and a scalar tail of < 1024 tokens; one call for
    all of a: parse 3.75 vs 4.1 ms in 32 calls) and fewer column-pass levels (32 rows spill).
  - No row skew: column pass 11 ms instead of 0.93 (2^15 rows; 64 lines in one cache set).
  - Lane bits: in-register shift levels (12 ops per vector) vs transposed tiles (3 shuffles per
    vector plus cross-vector levels): part of v1 → v2.
  - In memory, 2^20 values, forward row transforms over 17 bits: L1 pieces of 2^12 0.364 ms, of
    2^15 0.395; radix-16 sweeps 0.73 (radix-8 kept). ~0.95 cycles per vector butterfly; the bound
    is ~0.75 (3 ops on 4 pipes, 8 stores per 12 butterflies).
  - Products, 2^20 values in memory: Montgomery + Shoup 0.343 ms (kept); Shoup with a quotient of
    y from doubles 0.402; Montgomery twice (by R^2) 0.390; Shoup with the constant products as
    shifts and adds 0.455.
  - Column pass with the next column's forward levels software-pipelined under the products:
    0.522 vs 0.551 ms in memory. Not kept (0.2% of the total).
  - `std::_Exit(0)` after the flush (skips the input `munmap`; exit unmaps it instead): 13.79 vs
    13.74 ms. Not kept.
- 2026-10-09, claude: submitted the round-1 `main.cpp` (PR #38),
  [409181](https://judge.yosupo.jp/submission/409181): AC, 23 ms (1/5). Cases at N = 20: 12, 23,
  12 ms; `lc-amd` measures 13.7 ms for the slowest of the three. Not resubmitted for the outlier.
- 2026-10-09, claude: resubmitted the same `main.cpp` at the user's request,
  [409186](https://judge.yosupo.jp/submission/409186): AC, 13 ms, 21.2 MiB (2/5).
- 2026-10-09, claude, round 2 (v2 → v3). All on `lc-amd`, judge image and flags; "probe" means
  whole-process wall time and in-process phase stamps on max_random_01, runs interleaved, medians.
  - Barrett product: 1.475 ns per vector against 2.67 for Montgomery + Shoup (in memory, 2^12 and
    2^20 values, checked on 2 * 10^7 random and edge pairs); low word of x y from `vpmulld` instead
    of shift and blend: 1.375. Kept. Column pass 0.55 → 0.38 ms.
  - `.preinit_array` start, `_exit`, and a/b in 4 huge pages (the 1 KiB beyond 8 MiB used to fault
    a fifth): with Barrett, probe total 12.13 → 11.82 ms; Barrett alone 12.09.
  - Sweeps skip the store of the unchanged last vector: forward rows 0.748 → 0.726 ms, inverse
    0.402 → 0.393.
  - Upper radix-8 row sweep per 128 KiB group of pieces (while in L2): forward 0.722 vs 0.726,
    inverse 0.422 vs 0.393. Dropped.
  - Read calls (probe, 61 runs, total ms): 16 (one per skewed row) 11.70; 2 (each array
    contiguous, b 2 KiB after a) 11.57; 4 (bands of 4 rows, 1 KiB skews) 11.61; 8 (bands of 2)
    11.63; 2 with b 34 KiB after a 11.64. Parse 3.76 → 3.64; the column pass did not suffer
    (0.37 vs 0.39). Without huge pages (madvise removed; 41 runs): 16 calls 15.66, 2 calls 15.47,
    4 calls 15.23. Kept bands of 4: equal with huge pages, best without.
  - Without huge pages a/b fault in 4 KiB pages: parse 4.03 → 7.0 ms, exit 0.21 → 0.68
    (~1.45 µs per 4 KiB fault on this VM).
  - Input alone (20.7 MB, every line touched, probe): mapped 1.48 + `munmap` 0.68 ms; `read()`
    into a reused buffer of 64 KiB 1.87 ms (256 KiB 1.92, 1 MiB 2.39). A streamed Reader would
    save ~0.25 ms here (input read before any output); that is lib/io's (#21).
  - Checks: 13/13 official tests; `stress.py` 120 rounds; ASan/UBSan on all 13 official tests,
    file and pipe input.
  - PR #48 merged. CI ratios: Xeon 8370C 0.929, EPYC 7763 0.964, EPYC 9V74 0.941 (geomean 0.944).
- 2026-10-09, claude: submitted the round-2 `main.cpp` (PR #48),
  [409202](https://judge.yosupo.jp/submission/409202): AC, 12 ms, 19.0 MiB (3/5). max_random_00,
  _01, _02: 12 ms each (round 1: 12-13 ms).
- Next (round 2): the transforms (1.5 ms) run at ~3 vector ops per butterfly plus one store per
  element per pass, near the 4-pipe bound; little left there. Remaining time is lib/io (parse
  2.5 ms, input faults and `munmap` 1.6 ms), `../fixed_width.hpp` (1.13 ms) and `write()` (3.4 ms).
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  13.17 → 12.79 ms (0.979). 13/13 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib/io #21, round 3): submitted the #176 `main.cpp`,
  [409382](https://judge.yosupo.jp/submission/409382): AC 21 ms, 19.0 MiB; one launch spike
  (max_random_01 21, peers 12), clean 12 ms (`tools/spikes.py`). Best judged stays 12 ms.
- 2026-10-10, claude (lib/io #21, round 4): the output text goes into b, dead after the product
  and already touched, page-aligned, instead of `fixed_width.hpp`'s static 250 KB (63 page faults;
  497 → 428 faults per run), and blocks are 60 pages (`lib/io/notes.md`: whole-page writes from
  a page-aligned buffer are the fastest). `judge.py bench`, `lc-amd`, 31 rounds, slowest 3 cases:
  12.93 → 12.66 ms (0.985). First try: the buffer move alone 0.988, with 60-page blocks 0.983.
  Outputs byte-identical to main on all 13 tests (judge build, ASan/UBSan, pipe input).
  PR #187 merged; CI 0.9882 (EPYC 7763 0.982, EPYC 9V45 0.990, Xeon 8370C 0.993).
- 2026-10-10, claude (lib/io #21, round 4): submitted the #187 `main.cpp`,
  [409404](https://judge.yosupo.jp/submission/409404): AC 21 ms, 18.8 MiB. max_random_00 and
  _02 12 ms; max_random_01 21, as in 409382 (12 in 409202): a launch spike twice on the same
  case (locally the three cases are within 0.2 ms). Clean 12 ms; best judged stays 12 ms.
- 2026-10-10, claude (lib/io #21, round 5): output through `../convolution_mod/fields.hpp` (16
  values per step; in memory 0.64 ms per 2^20 values against 1.04 for `fixed_width.hpp`,
  `../convolution_mod/notes.md` round 2) instead of `fixed_width.hpp`, now deleted. Text still
  page-aligned in b (`../text_buffer.hpp`); 256000-byte blocks instead of 60 pages (whole pages
  gain nothing, `lib/io/notes.md` round 5). `judge.py bench`, `lc-amd`, 31 rounds, slowest 3
  cases: 12.56 → 12.27 ms (0.973). Outputs byte-identical to main on all 13 tests and 200
  random inputs (N 0-13), also with `-DBLOCK_LOG=6`; ASan/UBSan on all 13 tests, file and pipe.
  PR #198 merged; CI 0.9693 (EPYC 9V74 0.967, EPYC 9V45 0.984, Xeon 8573C 0.958). Submitted
  [409430](https://judge.yosupo.jp/submission/409430): AC 12 ms, 18.8 MiB, no spike. Best judged
  stays 12 ms (the gain is 0.3 ms).
- 2026-10-10, claude, round 3: no gain; code unchanged. `lc-amd`, judge image and flags,
  max_random_01, medians. Scripts, sources and raw numbers: `lc-opt-explore/bitwise_and_convolution/`
  (`results.txt`).
  - Where the time goes. In-process (31 runs): parse 3.30 (input faults ~0.96 and a/b huge pages
    ~0.23 included), forward rows 0.72, column pass 0.39, inverse rows 0.39, format 0.65,
    `write()` 3.40, input `munmap` 0.68; total 9.58. Outside it (41 runs, own runner): fork to
    `.preinit_array` 1.03, exit 0.19. Page faults: 104 before preinit, 339 in the parse, 7 after.
  - Startup: empty programs, 300 runs: `g++` 1.13 ms, `g++` with the preinit `_exit` 0.965, `gcc`
    0.549, `-Wl,--as-needed` 0.555, `-static` 0.384. Loading libstdc++, libm and libgcc_s costs
    ~0.41 ms; the judge's link line adds them, so the source cannot avoid it.
  - `tools/runner.c` opened the output with `O_TRUNC`, and `judge.py` reuses each output path, so
    each timed run also frees the previous run's 10.5 MB output: fork to preinit 2.27 ms against
    1.03 with the file unlinked first (whole run 12.01 → 10.74). A constant for this problem; it
    explains why `judge.py bench` (12.3) read above the judge (12). Fixed in #221: `runner.c`
    now unlinks the output before the clock starts.
  - Zen 3 vector pipes (12 independent chains): add, sub, min, or, `vpblendd` 4 per cycle;
    multiplies (`vpmuludq`, `vpmulld`, `vpmulhuw`, `vpmullw`, `vpmaddwd`) and `vpblendvb` 2 per
    cycle on one pipe pair; `vpshufb`, unpacks and immediate shifts 2 per cycle on the other.
  - Transform work inside the parser loop (a copy of `BulkParser32`'s lockstep with one radix-8
    or radix-4 group per step; 2^20 tokens in memory): parse 0.99 ms, groups alone 0.46 / 0.26,
    fused 1.65 / 1.24. Slower than the sum: the parser leaves no vector slots free. Dropped.
  - Inverse tiles fused with the text (each tile's 8 vectors formatted from registers), 2^20
    values in memory: 1.036 ms against 0.806 separate; tile i+1 before the text of tile i 0.838;
    per 4096-value piece 0.810. Dropped.
  - Formatter (`../convolution_mod/fields.hpp`, 0.647 ms per 2^20 values): 28 ops per 16 values
    on the multiply pair, 33 on the shuffle/shift pair, 38 on any pipe; ~25 cycles at full issue,
    34.4 measured. Variants: 32 values per step 0.681; odd lanes by offset loads instead of
    shifts 0.655; `<< 8` as a multiply 0.647; text without `vpblendvb` 0.658; combined
    0.664-0.670. None kept.
  - Column pass with software prefetch 16, 64, 256 vectors ahead: 0.48-0.52 ms against 0.36.
    Input unmapped before the output: `write()` 3.40 vs 3.43 ms. Pieces of 2^11 / 2^13 values:
    forward rows 0.733 / 0.769 ms against 0.728 (in memory). None kept.
  - Transforms at ~77-80% of the 4-pipe issue bound (L1 pieces 0.49 ms per 2^21 values against
    ~0.40; column pass 0.35 against ~0.27, estimates from op counts); upper row sweeps (in L2)
    ~0.1 ms above their ALU time. The gap to a perfect schedule is ~0.45 ms over all transforms;
    no tested change closed any of it.
  - Next: the kernel holds ~6.5 of ~10.7 ms (start 1.03, input faults and `munmap` 1.64, a/b
    zeroing 0.23, `write()` 3.40, exit 0.19), user code ~4.3 (parse 2.1, transforms 1.5, format
    0.65). Untried: `MADV_SEQUENTIAL` on the input (-0.04 ms, `lib/io/notes.md` round 4; belongs
    in lib/io); the upper 5 row bits in the column pass (8 bits per column block, in an L1
    scratch) instead of L2 sweeps: at most ~0.15 ms (a guess), and it needs a layout skewed per
    piece, which costs many `read_bulk` calls.
- 2026-10-10, claude, round 4 (`agent/bitwise_and_convolution-r4`). Judge flags and image;
  builds on `lc-amd`, timing on `lc-bench` (EPYC 7B13, idle). "wall" is fork to exit with the
  output unlinked first, interleaved runs on max_random_01, median of paired ratios; phases are
  in-process stamps. Sources, scripts, raw numbers: `lc-opt-explore/bitwise_and_convolution/r4/`.
  - Judge: 409430 scored max_random_00/01/02 at 11/12/11 ms, so ~0.2 ms could make 11.
  - Kept, input and output: `madvise(MADV_SEQUENTIAL)` on the input mapping (from
    `Reader::scan().cur`; `munmap` 0.674 → 0.635 ms), and printing per band in whole 25600-value
    blocks (each row used to end with 3072 values, 30 KB, copied into the Writer's stack buffer
    and flushed by the next block). Both: wall 0.9824 (61 runs); `judge.py bench` 0.9939 (31).
  - Kernel costs in core cycles per 8 vectors (microbenchmark `xb.cpp`): radix-8 sweep in L1 9.4
    (ALU bound 9); tile (levels, transpose, levels) 32.0 (bound 24); upper sweeps of a row in L2
    21.9 (ALU 15); column pass 76 per column warm (bound 57), 134 from memory.
  - Zen 3 pipes: every timing fits "shuffles on 2 pipes, other ops split evenly over 4": tile
    (24 + 72 / 2) / 2 = 30 (32.0); transpose + levels (24 + 36 / 2) / 2 = 21 (19.5-20.1).
  - Tile variants (`xt.cpp`, piece in L1, forward / inverse): one pass 32.0 / 32.5; sweep at
    stride 1 then transpose + levels 29.4 / 27.6; one pass software-pipelined over 2 tiles 33.0 /
    33.2; 128-bit stage by `vinserti128` loads 28.9 / 28.0; by 128-bit broadcasts and `vpblendd`
    27.6 / 27.3. Kept the last.
  - In the program the two-pass tiles first lost: forward rows 0.803 vs 0.749 ms (0.746 vs 0.738
    in a second run). The forward's first pass over a piece reads it from L3 (just parsed); a
    9.4-cycle sweep cannot hide that, the 32-cycle tile could. Prefetching the next piece during
    the transpose pass: 0.711 / 0.707 (T0 / T1 hint) vs 0.738-0.753. The inverse (pieces in L2
    after its upper sweeps): 0.393-0.403 vs 0.397-0.402 with or without prefetch; prefetch kept
    for the forward only.
  - Column pass software-pipelined (forward levels of column j + 1 beside the product of column
    j, through an L1 buffer): 73.5 vs 77.9 cycles per column in `xc.cpp`, but 0.398 vs 0.364 ms in
    the program. Dropped.
  - Upper sweeps: one radix-32 pass through a 1 KiB L1 scratch (32 vectors 16 KiB apart): 10x
    slower (all in one L1 set). Software prefetch 8-64 vectors ahead: 25.2-27.6 vs 23.7. Dropped.
  - S8, every 2^12 piece skewed by a cache line and all 8 bits above the piece in the column
    pass through an L1 scratch (no upper row sweeps; pieces moved to the skewed layout by the
    forward piece pass): byte-identical output, but transforms 1.574 vs 1.509 ms: column pass
    0.77 ms. Per unit (TSC): gathers fill 64-byte lines for 32 bytes each (even units also miss
    L3), the product stage spills. With the top radix fused into the product: same; with
    prefetch of the next unit's lines: 0.81. Dropped.
  - Final (`main.cpp` of this round) against round 3's: `judge.py bench`, 41 rounds, slowest 3
    cases: 11.90 → 11.83 ms (0.9902). Checks: 13/13 official tests; `stress.py` 120 rounds; tokens
    equal to round 3's output on the 13 official and 57 generated inputs (N = 0..20), file and
    pipe, also with ASan/UBSan. Split tiles on `lc-intel` (Emerald Rapids, x86-64-v3, `xt.cpp`):
    also faster, 137 vs 153 (forward) and 133 vs 154 (inverse), relative units.
  - PR #233 merged; CI 0.9916 (EPYC 7763 0.9925, EPYC 9V45 0.9937, EPYC 9V74 0.9885).
- 2026-10-10, claude: submitted the #233 `main.cpp`,
  [409492](https://judge.yosupo.jp/submission/409492): AC 11 ms, 18.8 MiB (1/5 this round);
  max_random_00, _01, _02 11 ms each, no spike (`tools/spikes.py`). New best (was 12 ms).
- Next: the kernel still holds ~6.5 of ~10.7 ms (start, input faults and `munmap`, `write()`).
  User side, in core cycles: column pass 76 per column against 57 (the product's 6 multiplies
  and 5 shifts per vector), upper row sweeps 21.9 per 8 vectors against 15 (L2), split tiles
  27.5 against ~24; the formatter (`fields.hpp`) 34 cycles per 16 values against ~26.
- 2026-10-10, claude (lib, issue #156 round 3): `advise_sequential` comes from
  `lib/io/sequential.hpp` (`io::advise_sequential`) instead of a local copy. Same stripped
  executable as before (judge flags, `lc-amd`).
- 2026-10-10, claude (lib, issue #156 round 3): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).

## Sources

- Fast zeta and Mobius transforms over the subset lattice: standard; written from the definition.
- Montgomery multiplication: P. Montgomery, "Modular multiplication without trial division",
  Math. Comp. 44 (1985). Shoup multiplication by a constant with a precomputed quotient: as used in
  NTL and in `lib/ntt` (`lib/ntt/notes.md`). Both written here from the formulas (round 1).
- Barrett reduction: P. Barrett, "Implementing the Rivest Shamir and Adleman public key encryption
  algorithm on a standard digital signal processor", CRYPTO '86. Shift and error bound derived here.
- `.preinit_array` start: taken from `../convolution_mod/solution.cpp`.
- 8x8 transpose of 32-bit lanes with unpack/permute2x128: the common AVX2 idiom. The 128-bit stage
  by broadcast loads and blends: derived here (round 4), as `fields.hpp` loads its value pairs.
