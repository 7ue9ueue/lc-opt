# lib/mem

Zero-filled memory in transparent huge pages. API and usage: the headers of `huge.hpp` and
`write_first.hpp`. Tests: `lib/mem/test.cpp`.

## Design

- `map_huge(bytes)`: an anonymous private mapping of the size rounded up to 2 MiB plus one more
  huge page, its first 2 MiB boundary returned, `MADV_HUGEPAGE` on the rounded size. Nothing is
  touched: pages fault in on first write, so a huge page is zeroed only when used. Never unmapped;
  the programs end with `_exit`.
- `huge<T>(count)`: `map_huge` for count values of T.
- `Arena`: one `map_huge` mapping, bump allocation in 64-byte steps, no bounds check (callers size
  it from the same counts they take).
- Why huge pages: 4 KiB pages cost 0.8 ms per 2 MiB in faults on `lc-amd`, a huge page 0.04-0.11
  ms (`problems/convolution/convolution_mod/notes.md`, round 3); floors drop 15-20% (`lib/io/notes.md`).
- `write_first(x, begin, end)`: one store of zero at each 2 MiB boundary in x[begin, end). On
  Linux 6.8 (`lc-k68`, whose times match the judge's) a huge page that is read before it is
  written maps the shared huge zero page, and the first write then splits it into 4 KiB pages;
  Linux 7.0 (`lc-amd`) allocates a huge page either way. convolution_mod_large on `lc-k68`:
  small_and_large_01 340.2 -> 310.3 ms, random_02 381.2 -> 369.6
  (`problems/convolution/convolution_mod_large/notes.md`, round 3). Its own header, so that the
  problems that include `huge.hpp` keep their `main.cpp`.

## Users

| Problem | Before | Now |
|---|---|---|
| gcd_convolution, min_plus_convolution_concave_arbitrary | `allocate<T>(count)` | `huge<T>` |
| lcm_convolution | `allocate(bytes)` | `huge<char>` |
| min_plus_convolution_convex_arbitrary, min_plus_convolution_convex_convex, convolution_F_2_64 | `allocate(words)` | `huge<u32>`, `huge<u64>` |
| convolution_mod_1000000007, convolution_mod_2_64, multivariate_convolution_cyclic | `Arena` | `Arena` |
| `problems/convolution/floor.cpp` (I/O floor harness) | `allocate(count)` | `huge<Value>` |

`write_first`: convolution_mod_large (its origin). It may pay wherever a transform reads zero
tails that span whole 2 MiB pages before anything writes them (a guess; for problem rounds to
measure on `lc-k68`): convolution_mod_2_64 (a and b are 2 huge pages each at the largest size;
a short factor's second page is read by the first level and later written as the last prime's
work), and products in lib/poly or lib/ntt with one short factor. Not convolution_mod or
convolution_mod_1000000007: their factors' lower halves are one huge page each, which the parser
writes first.

Not moved, because they map differently (a change of behaviour, not a move):
- bitwise_and_convolution, bitwise_xor_convolution: whole huge pages plus a remainder under 1 MiB
  in small pages just below them (bitwise_and's round is running).
- multivariate_convolution: arrays under 256 KiB in small pages.
- mul_mod2n_convolution: an arena with a bounds check and padded word arrays.
- `ntt::Convolution`, `ntt::Product`, mul_modp_convolution's and the polynomial problems' products:
  they unmap in their destructors (lib/ntt, lib/poly).

## Log

2026-10-10, claude (issue #156, round 2): extracted from 9 convolution problems, which had the
same mapping in three forms (`allocate<T>`, `allocate(words)` with the extra page counted first,
and a bump arena).
- Compiled code (judge flags, GCC 15.2 image, `lc-amd`), against main: the stripped executables
  are byte-identical for gcd, lcm, min_plus convex_arbitrary, min_plus convex_convex,
  convolution_F_2_64 and convolution_mod_2_64. convolution_mod_1000000007 and
  multivariate_convolution_cyclic: same instructions, other stack slots in `solve()` (and in
  cyclic's `transform_short`). min_plus concave_arbitrary: `allocate<T>` was out of line and is now
  inlined into `solve()`; `columns::detail::format<10>` keeps its instructions with other stack
  slots.
- `judge.py bench` on `lc-bench` (EPYC 7B13), 21 rounds, slowest 3 cases, the three that differ:
  convolution_mod_1000000007 23.53 -> 23.54 ms (ratio 1.0060), min_plus concave_arbitrary
  27.61 -> 27.78 (1.0009), multivariate_convolution_cyclic 11.50 -> 11.44 (1.0039). Noise: the
  ratios are within a session's spread of identical binaries (`lib/multimod/notes.md`: 0.25%;
  CI's per-job sd 0.13-1.9%).
- Checks: `test.cpp` at -O2 and with ASan/UBSan (`lc-amd`); official tests of all 9 problems pass;
  ASan/UBSan builds of the 3 that differ on their slowest and smallest official cases, file and
  pipe input.

2026-10-10, claude (issue #156, round 3): `write_first` from convolution_mod_large, generic over
the element type (the same instructions for `uint32_t`).
- Compiled code (judge flags, `lc-amd`): the stripped executable is byte-identical to main's.
  `write_first` is `static`: as a plain function template, `convolve` grew from 70 to 74
  instructions, keeping two values in callee-saved registers across the calls, because GCC's
  interprocedural register allocation sees which registers a local function clobbers, not a
  template that may be replaced at link time. That version: `judge.py bench` on `lc-bench`, 15
  rounds 407.83 -> 410.25 ms (1.0059), then sources swapped, 21 rounds, 406.30 (new) against
  408.28 (old); CI 1.0005 and, re-run, 1.0024 (#343). Code alignment may explain a part (`.text`
  +71 bytes); 4 instructions run once per process.
- Checks (`lc-amd`): `test.cpp` (new: stores exactly at the boundaries in [begin, end), for
  `uint32_t` and `uint64_t`, four alignments, eight ranges) at -O2 and with ASan/UBSan; rounding
  the first boundary down instead of up fails it. convolution_mod_large: 54/54 official tests.

## Sources

- The problems' former copies (git history; their notes list their measurements).
