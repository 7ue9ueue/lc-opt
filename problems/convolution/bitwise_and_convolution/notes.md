# bitwise_and_convolution

N <= 20; 2^N values a_i, b_i < 998244353; print c_k = sum over i & j = k of a_i b_j, mod 998244353.
5 s. Inputs are ~20.7 MB, outputs ~10.5 MB (fixed width) at N = 20.

Best judged: ours, [409186](https://judge.yosupo.jp/submission/409186), 13 ms (first). Same file as
409181 (23 ms), whose 23 ms max_random_01 was judge noise.
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
  the inverse row bits; each row then gets its inverse 17 bits and is printed with
  `../fixed_width.hpp`.
- Inside a row: pieces of 2^12 values (16 KiB, L1), then the row's upper bits. The lane bits use a
  transpose: per 8-vector tile, radix-8 over the vector bits, 8x8 transpose, radix-8 over the former
  lane bits. Forward tiles stay transposed (the product does not care); the inverse restores them.
  Sweeps do not store the last vector of a group (all bits set: no level changes it).
- Shorter inputs are padded with zeros to 2^6 values; zeros do not change c_k for k < 2^N.
- Memory: a and b in one mapping, 4 huge pages (`MADV_HUGEPAGE`) and one 4 KiB page below them
  for the 4 KiB beyond 8 MiB.
- Runs from `.preinit_array` and ends with `_exit` (as `convolution_mod`): libstdc++'s
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
- Next: the transforms (1.5 ms) run at ~3 vector ops per butterfly plus one store per element
  per pass, near the 4-pipe bound; little left there. Remaining time is lib/io (parse 2.5 ms,
  input faults and `munmap` 1.6 ms), `../fixed_width.hpp` (1.13 ms) and `write()` (3.4 ms).

## Sources

- Fast zeta and Mobius transforms over the subset lattice: standard; written from the definition.
- Montgomery multiplication: P. Montgomery, "Modular multiplication without trial division",
  Math. Comp. 44 (1985). Shoup multiplication by a constant with a precomputed quotient: as used in
  NTL and in `lib/ntt` (`lib/ntt/notes.md`). Both written here from the formulas (round 1).
- Barrett reduction: P. Barrett, "Implementing the Rivest Shamir and Adleman public key encryption
  algorithm on a standard digital signal processor", CRYPTO '86. Shift and error bound derived here.
- `.preinit_array` start: taken from `../convolution_mod/solution.cpp`.
- 8x8 transpose of 32-bit lanes with unpack/permute2x128: the common AVX2 idiom.
