# convolution_mod_large

N, M <= 2^24 coefficients mod 998244353; print the N + M - 1 coefficients of the product. 10 s,
1 GiB (the output file's tmpfs pages count). Inputs and outputs are ~331 MB at the maximum.

Best judged: ours, [409233](https://judge.yosupo.jp/submission/409233), 448 ms (current `main.cpp`).
Before: [408888](https://judge.yosupo.jp/submission/408888), 452 ms, the QPoly exploration-014
program (`../SymPoly/work/ntt/yosupo_convolution_mod_large_opt.cpp`, guess from the submission dates). Next other user: 737 ms (403499). Judge phases of 406521 (the exploration-011
twin, 454 ms): parse 71, NTT 205, output 156 (`write()` ~120), ~12-15 outside `main`.

## Design

`lib/ntt` (transform length 2^25 at the maximum), `lib/io` input, output by
`../convolution_mod/fields.hpp` (same bytes as `../fixed_width.hpp`), start from `.preinit_array`
and `_exit` as in `../convolution_mod`.

## Log
- 2026-10-09, claude: refactored the QPoly program onto `lib/ntt`, `lib/io` input and
  `../fixed_width.hpp` output. 54/54 official tests (`tools/judge.py` now runs large problems in
  tmpfs-sized batches); stress test 100 rounds. `lc-amd`, 11 rounds, slowest 3 cases: 424.6 ms vs
  426.7 for the QPoly exploration-014 program, ratio 0.994.
  Phases on fft_killer_04 (ms): parse 64.4 vs 64.9, transform 201.4 vs 202.6, output 144.3 vs
  142.7, exit 15.
- Output path, same case: formatter out of line (GCC did not inline it) 177.5 ms; inlined 166;
  constants through an opaque pointer (memory operands instead of rebuilt broadcasts) and fewer
  operations 156.9; 250 KB blocks passed to `io::Writer::write(std::string_view)`, which writes
  them directly (one `write()` per block instead of per 64 KiB) 144.3 (QPoly: 142.6).
  In memory, 2^25 values: 48.7 → 34.7 ms (QPoly asm: 32.3).
- Tried: a 256 KiB `io::Writer` as a template capacity, 147.6; it moved functions in aplusb's
  binary and CI blocked it (see `lib/io/notes.md`). The block writes are faster anyway.
- Tried and dropped: `std::_Exit(0)` after the flush (427.7 vs 427.8 ms); one table copy per use
  of each formatter constant (34.1 vs 34.7 ms in memory, ~15 more lines).
  Not submitted yet: 0.6% is below the judge's spread (452-461 ms over our recent submissions).
- 2026-10-09, claude: submitted the current `main.cpp`. [409233](https://judge.yosupo.jp/submission/409233):
  AC, 448 ms, 612.7 MiB (1/5), large cases 440-448. [409236](https://judge.yosupo.jp/submission/409236):
  AC, 453 ms (2/5), large cases 441-453. Judged runs sit 4-7% above `lc-amd` (424.6 ms) on every
  large case, as the QPoly program did (426.7 vs 452): a systematic offset, not jitter. A guess: page
  faults of the ~613 MiB footprint cost more on the judge. Not resubmitted further.
- 2026-10-09, claude (round 2). `lc-amd`, judge flags; phases from in-process `CLOCK_MONOTONIC`
  stamps on fft_killer_04 (scratch probe, not committed), 5 runs, ms:
  - Before: parse 61.5 (of which input page faults ~20 and zero-filling a and b's lower halves
    8.4; parse alone 41.6, 1.24 ns per token), tables 2.9, top radix-4 of a and b 12.8 (mostly
    zero-filling the upper halves), four quarters 178.5, inverse top 6.8, output 132 (format ~22,
    `write()` ~110). Exit: unmapping the input alone takes 12.
  - Per radix-4 level (forward a and b, inverse a): 9.1 + 4.5 at every size from 2^10 to 2^18
    vectors, 10.8 + 4.2 at 2^20. Memory is not the limit: fusing top levels cannot gain more than
    ~2 ms. The `forward` kernel issues ~14.6 vector ops per vector and level; 3.76 cycles measured
    against 3.66 at 4 ops per cycle. Tiles (levels below 2^10 vectors, leaf products) 94.4.
  - Kept: `fields.hpp` formatter and `.preinit_array` start. `tools/judge.py bench`, 11 rounds,
    slowest 3 cases: 428.3 -> 414.6 ms, ratio 0.974 (second run: 427.1 -> 414.8, 0.968).
  - Lost: input by 256 KiB `read()` chunks into an L2 buffer, parsed by `BulkParser` (no input
    mapping, no faults, no unmap; footprint 450 -> 293 MiB): 418.8 ms vs 414.8, ratio 0.985 vs
    0.968. Alone, reading the file costs 34 ms by `read()` against 42 by mmap + touch + unmap,
    but the parse then loses the overlap with its DRAM reads.
  - Lost: `MADV_POPULATE_READ` on the input mapping: 26.5 ms vs 22 for faulting it by touch.
  - Checks: 54/54 official tests, stress 200 rounds, ASan/UBSan on 6 official cases (incl.
    max_random_00) and pipe input.
  - Next: the transform is at the vector ALU bound of its kernels (~190 of ~415 ms); gains need
    fewer ops per butterfly or cheaper leaves (`lib/ntt`). Kernel time (input faults 20, zero
    fill 16, `write()` 110, unmap ~15) is ~160 ms.
