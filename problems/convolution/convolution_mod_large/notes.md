# convolution_mod_large

N, M <= 2^24 coefficients mod 998244353; print the N + M - 1 coefficients of the product. 10 s,
1 GiB (the output file's tmpfs pages count). Inputs and outputs are ~331 MB at the maximum.

Best judged: ours, [408888](https://judge.yosupo.jp/submission/408888), 452 ms: the QPoly
exploration-014 program (`../SymPoly/work/ntt/yosupo_convolution_mod_large_opt.cpp`, guess from the
submission dates). Next other user: 737 ms (403499). Judge phases of 406521 (the exploration-011
twin, 454 ms): parse 71, NTT 205, output 156 (`write()` ~120), ~12-15 outside `main`.

## Design

Same program as `../convolution_mod`: `lib/ntt` (transform length 2^25 at the maximum), `lib/io`
input, `../fixed_width.hpp` output.

## Log
- 2026-10-09, claude: refactored the QPoly program onto `lib/ntt`, `lib/io` input and
  `../fixed_width.hpp` output. 54/54 official tests (`tools/judge.py` now runs large problems in
  tmpfs-sized batches); stress test 100 rounds. `lc-amd`, 9 rounds, slowest 3 cases: 429.4 ms vs
  426.5 for the QPoly exploration-014 program, ratio 1.006.
  Phases on fft_killer_04 (ms): parse 65.0 vs 65.6, transform 200.8 vs 202.5, output 151.1 vs
  142.5, exit ~16 both.
- Output path, same case: formatter out of line (GCC did not inline it) 177.5 ms; inlined 166;
  constants through an opaque pointer (memory operands instead of rebuilt broadcasts) and fewer
  operations 156.9; `io::BasicWriter` with 256 KiB instead of 64 KiB 151.1.
  In memory, 2^25 values: 48.7 → 34.7 ms (QPoly asm: 32.3).
- Tried and dropped: `std::_Exit(0)` after the flush (427.7 vs 427.8 ms); one table copy per use
  of each formatter constant (34.1 vs 34.7 ms in memory, ~15 more lines).
  Not submitted: not faster than the judged 452 ms.
