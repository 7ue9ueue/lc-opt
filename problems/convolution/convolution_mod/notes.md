# convolution_mod

N, M <= 2^19 coefficients mod 998244353; print the N + M - 1 coefficients of the product. 5 s.

Best judged: ours, [408716](https://judge.yosupo.jp/submission/408716), 14 ms: the QPoly
exploration-011 program (`../SymPoly/work/ntt/yosupo_convolution_mod_large_io_probe.cpp`, guess
from the submission times and `lib/io/notes.md`). Next other user: 23 ms (393435).

## Design

- `lib/ntt`: one cyclic transform of length 2^lg >= N + M - 1 (2^20 at the maximum).
- `lib/io` for input (bulk `uint32_t` read straight into the transform buffers).
- Output: `../fixed_width.hpp`, every value in a 10-byte field (judge-specific; the checker
  compares tokens).

## Log
- 2026-10-09, claude: refactored the QPoly program onto `lib/ntt`, `lib/io` input and
  `../fixed_width.hpp` output. 53/53 official tests; stress test 400 rounds (`../stress.py`).
  `lc-amd`, 31 rounds, slowest 3 cases: 14.44 ms vs 14.73 for the 408716 source (QPoly
  `yosupo_convolution_mod_large_io_probe.cpp`), ratio 0.979. Phases at N = M = 2^19 (ms):
  parse 1.9, transform 4.7, output 4.8 (`write()` ~3.7), exit 0.4, start ~1.9.
  Not submitted yet: 2% is about one judge tick (14 ms judged).
- Idea, not tried: radix-8 first and last levels (2^20 has a radix-2 top), saving the copy and
  scale passes over 4 MiB arrays. QPoly's copy-free top level lost 10% at this size (cache-set
  conflicts between streams 512 KiB apart, a guess), so it needs a layout that avoids them.
