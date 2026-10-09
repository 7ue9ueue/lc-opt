# bitwise_and_convolution

N <= 20; 2^N values a_i, b_i < 998244353; print c_k = sum over i & j = k of a_i b_j, mod 998244353.
5 s. Inputs are ~20.7 MB, outputs ~10.5 MB (fixed width) at N = 20.

Best judged: none yet. Record when the issue opened: 26 ms.

## Design

- c = mu(zeta(a) * zeta(b)): zeta sums over supersets (x[i] += x[i | bit] for each bit), mu
  inverts it (x[i] -= x[i | bit]). 3 transforms of 20 levels and 2^20 products at the maximum.
- Arithmetic: values in [0, P). One butterfly is add, subtract P, unsigned min (3 vector ops); the
  inverse is subtract, add P, min. Product: Montgomery (x y 2^-32, 6 `vpmuludq`) then a Shoup
  multiply by 2^32 mod P (2 `vpmuludq`, 2 `vpmulld`).
- Layout: rows of 2^17 values (8 rows at N = 20), each row 16 words (one cache line) longer. Each
  row is read with one `Reader::read` call, then transformed over its own 17 bits; one column pass
  then does the 3 row bits of a and b, the product and the inverse row bits; each row then gets its
  inverse 17 bits and is printed with `../fixed_width.hpp`.
- Inside a row: pieces of 2^12 values (16 KiB, L1), then the row's upper bits. The lane bits use a
  transpose: per 8-vector tile, radix-8 over the vector bits, 8x8 transpose, radix-8 over the former
  lane bits. Forward tiles stay transposed (the product does not care); the inverse restores them.
- Shorter inputs are padded with zeros to 2^6 values; zeros do not change c_k for k < 2^N.
- Memory: a and b in one 2 MiB-aligned mapping with `MADV_HUGEPAGE`.

## Measurements

`lc-amd` (EPYC 7B13), GCC 15.2 in the judge image, judge flags, `tools/judge.py bench`, slowest 3
cases (all max_random, N = 20), 21 rounds, 2026-10-09, branch `agent/bitwise_and_convolution`.

| Program | Median ms | Ratio |
|---|---|---|
| v1: rows of 2^15, lane bits by in-register shifts | 14.60 | 1 |
| v2 (this `main.cpp`): rows of 2^17, transposed tiles, L1 pieces | 13.74 | 0.942 |
| floor: same allocation, reads and output, no transforms | 11.91 | 0.874 of v2 |

Phases of v2 inside `main` (ms, max_random_01, 9 runs): parse 3.9, row transforms 0.75, column
pass 0.55, inverse row transforms 0.40, output 4.6 (`write()` ~3.5), input `munmap` 0.7. About 2.8
more outside `main` (start, exit). `perf` on `lc-intel` (static build): 40% of cycles in the kernel
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
- Next: the transforms cost 1.7 ms above the floor (3.9 M vector butterflies and 2^17 vector
  products at ~3.2 ops per cycle); hand-scheduled kernels might save ~0.3 ms (guess). The rest is
  I/O: parse (lib/io, #21) and the kernel's page work for the output file and input mapping.

## Sources

- Fast zeta and Mobius transforms over the subset lattice: standard; written from the definition.
- Montgomery multiplication: P. Montgomery, "Modular multiplication without trial division",
  Math. Comp. 44 (1985). Shoup multiplication by a constant with a precomputed quotient: as used in
  NTL and in `lib/ntt` (`lib/ntt/notes.md`). Both written here from the formulas.
- 8x8 transpose of 32-bit lanes with unpack/permute2x128: the common AVX2 idiom.
