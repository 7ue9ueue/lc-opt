# convolution_mod_large

N, M <= 2^24 coefficients mod 998244353; print the N + M - 1 coefficients of the product. 10 s,
1 GiB (the output file's tmpfs pages count). Inputs and outputs are ~331 MB at the maximum.

Best judged: ours, [409233](https://judge.yosupo.jp/submission/409233), 448 ms (current `main.cpp`).
Before: [408888](https://judge.yosupo.jp/submission/408888), 452 ms, the QPoly exploration-014
program (`../SymPoly/work/ntt/yosupo_convolution_mod_large_opt.cpp`, guess from the submission dates). Next other user: 737 ms (403499). Judge phases of 406521 (the exploration-011
twin, 454 ms): parse 71, NTT 205, output 156 (`write()` ~120), ~12-15 outside `main`.

## Design

Same program as `../convolution_mod`: `lib/ntt` (transform length 2^25 at the maximum), `lib/io`
input, `../fixed_width.hpp` output.

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
