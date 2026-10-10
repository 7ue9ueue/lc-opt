# convolution_mod_2_64

N, M <= 2^19 coefficients below 2^64; print the N + M - 1 coefficients of the product mod 2^64.
10 s. Record when opened: 76 ms (another user). Best judged: ours, 41 ms:
[409604](https://judge.yosupo.jp/submission/409604), [409605](https://judge.yosupo.jp/submission/409605)
(`main.cpp` of #294). Before: 43 ms, [409296](https://judge.yosupo.jp/submission/409296) (#124).

## Design

- Five NTT primes below 2^30 with 2^20 | p - 1: 998244353, 985661441, 976224257, 975175681,
  972029953. Product 2^149.35 > 2^19 (2^64 - 1)^2. One cyclic transform of length 2^lg per prime.
- `multimod::WideProduct` (`lib/multimod/wide_product.hpp`; this folder's `product.hpp` until
  #156 round 3; both factors at most half the transform, lg >= 9: all large tests): lib/multimod's
  transform (`Wide` 64-bit input, modulus a run-time value) with ntt::Product's bottom stage (two
  groups per kernel, no leaf weight array) and fused inverse top level. Its kernels
  (`lib/multimod/wide_kernels.hpp`) come from `gen_wide_kernels.py`: ntt::Product's asm with lib/multimod's modulus
  variables, and `forward_radix8_wide`, the first level for 2^lg = 2 * 4^j as a list-scheduled
  asm loop (lib/ntt's generator): it reads the 64-bit input and reduces it mod p on the fly
  (hi (2^32 mod p) + lo; lo < 8p brought below 2p by two halvings). Other sizes: lib/multimod.
- CRT, not Garner: the transform for prime k returns y_k = c / M_k mod p_k (the factor is folded
  into the final scale). c = sum y_k M_k - t M with t = floor(sum y_k / p_k); the fraction is
  c / M < 0.2, so t comes from a float sum + 0.4. Then c mod 2^64 from 64-bit constants
  C = lo + 2^32 hi: y lo by `vpmuludq` (even and odd lanes apart), y hi mod 2^32 by `vpmulld` in
  8 lanes. 18 multiplies per 8 values, no dependency chain between primes.
- Memory: a, b (64-bit), work, residues of primes 0-3: 7 arrays of 4 MiB at the largest size.
  The last prime writes its residues to work and uses a's storage as its work.
- Output (`fields64.hpp`, judge-specific): every value right-aligned in 20 characters after a
  space. top = x / 10^16, mid, low from multiply-shift estimates on the high bits, corrected once.
  mid and low -> 4-digit chunks -> digit bytes in 16-bit lanes; top's 4 characters from a table of
  1845; two stores per value (4 bytes of top, 16 of mid+low). Separators are never written: the
  buffer starts as spaces. Leading-zero logic for mid and low runs only when some top in the 8 is
  0. convolution_F_2_64 uses this formatter too.
- `.preinit_array` start and `_exit`, one huge-page arena (as convolution_mod).

## Log

- 2026-10-09, claude (round 1). `lc-amd`, judge flags, all_same_01 phases from in-process
  `CLOCK_MONOTONIC` stamps (scratch probe, not committed).
  - First version (5 primes, Garner, scalar SWAR formatter): 55.3 ms slowest case
    (`judge.py test`). Phases (ms): parse 2.5, reduce 2.5, transforms 5 x 4.5, Garner 5.0,
    format 11.6, `write()` 6.0. I/O floor (`floor.py`, plain write_array): 20.3 ms.
  - Garner constants hoisted (Shoup quotients were divided at run time): 5.0 -> 4.5. Direct CRT
    with float t: 2.3. Quotients of the next 8 issued before the products of these: 1.95.
  - Formatter, in memory, 2^20 values: scalar SWAR + digit-count mask 11.6; AVX2 with double
    estimates 6.0; 4-digit chunks in 16-bit lanes, top-zero fast path 6.2 (no change: latency
    bound, ~1.4 IPC); integer multiply-shift estimates instead of doubles 3.67; divisions of the
    next 8 values issued before the digits of these 2.62. Moving the chunk split into the first
    stage: 2.75 (worse).
  - Reduction fused into the radix-8 first level: per prime 0.46 + 4.54 -> 5.06 (no gain alone);
    with the cheaper reduction (two halvings instead of a Shoup product for lo, one permute) 4.98.
  - Now (ms): parse 2.5, transforms 5.3 + 4 x 5.0, CRT 1.95, format 2.6, `write()` 6.1.
    `judge.py test`, 3 runs: slowest 43.0, 42.8, 42.6 ms (all_same / gen_524288).
  - Checks: 44/44 official tests; stress 300 rounds (`stress.py`, lengths to 2^12 + 2, max
    coefficients); `test_fields64.cpp` 4.0M values against printf; ASan/UBSan stress 60 rounds and
    9 official cases (file and pipe input). UBSan caught `_mm_storeh_pd` to a misaligned double;
    replaced by `_mm_storel_epi64` of the high half.
  - Submitted the merged `main.cpp` (#114): [409249](https://judge.yosupo.jp/submission/409249)
    AC 54 ms from one outlier (gen_524288_00 54, the other large cases 43-44);
    [409250](https://judge.yosupo.jp/submission/409250) AC 45 ms. Matches `lc-amd` (42.8).
- 2026-10-09, claude (round 2). `lc-amd`, judge flags, all_same_01, in-process phases as round 1.
  Profile on `lc-intel` (`perf record`, whole process): 18% kernel time, `kernel_init_pages` 12%
  (zeroing fresh pages). Page cost on `lc-amd`: 32 MiB of huge pages first touched in 1.1-1.6 ms.
  Transform split (5 primes, ms): tables 0.28, radix-8 first level 4.0, subtrees 19.7, inverse
  top 0.56, final radix-2 and scale 0.62.
  - CRT high halves: y hi only matters mod 2^32, so one `vpmulld` in 8 lanes replaces two
    `vpmuludq`; even/odd lanes instead of `vpmovzxdq`. 1.96 -> 1.03 ms. Hot residues (CRT run
    twice) also took 1.94 before: it was compute-bound, not memory-bound. Kept.
  - Last prime: residues into work, its work in a's storage (a is read in full before work is
    written). 4 MiB fewer fresh pages; last prime 4.97 -> 4.77 ms. Kept.
  - Lost: -t M from two `vpermd` table lookups instead of three multiplies (t < 5): CRT 1.03 ->
    1.08. CRT fused into the formatter's loads (no block buffer): CRT + format 3.63 -> 5.36 (register
    pressure). Formatter with two independent 8-value groups per iteration: 2.60 -> 2.71.
  - Measured, not pursued: the radix-8 level's 64-bit reduction costs 1.2 of its 4.0 ms (a trivial
    stand-in: 2.88 -> 1.67 ms when hot); the arithmetic is ~17 ops per 8 values and no cheaper
    form was found (Montgomery 2^-32 variant: 16). Prefaulting everything: no net gain.
  - `tools/judge.py bench` (base = main): slowest 3 cases, 21 rounds 45.91 -> 44.74 ms
    median (ratio 0.973); 31 rounds 46.05 -> 44.84 (ratio 0.973). The VM was busier than in round 1
    (base 42.8 then).
  - Checks: 44/44 official tests; stress 300 rounds; ASan/UBSan stress 60 rounds and 5 official
    cases (file and pipe input).
  - Submitted the merged `main.cpp` (#124): [409260](https://judge.yosupo.jp/submission/409260)
    AC 44 ms; [409261](https://judge.yosupo.jp/submission/409261) AC 51 ms (judge jitter).
- 2026-10-09, audit (claude): resubmitted, [409296](https://judge.yosupo.jp/submission/409296) AC 43 ms
  (3/5), large cases 40-43 ms, no spike; `lc-amd` predicts ~41-43.
- 2026-10-10, claude (issue #156): the local transform moved to `lib/multimod` (`Wide`). At odd
  lg (2^7..2^19) a sparse input is now reduced and transformed in one pass; lg 20 is the same
  logic. Transforms alone at lg 20: ratio 1.0005; at lg 19: 0.9904. `judge.py bench`,
  31 rounds, `lc-amd`: 0.9998. 44/44 official tests. Details: `lib/multimod/notes.md`.
- 2026-10-10, audit (claude): [409346](https://judge.yosupo.jp/submission/409346) (2026-10-10
  01:48 UTC) was not logged before; who submitted it is not recorded. Current `main.cpp` (#162):
  AC 43 ms, 52.1 MiB. `tools/spikes.py` flags gen_265721_00 (43 ms, its peers 35), but
  all_same_01 and _03 also take 43: clean score 43. Ties best judged (409296).
- Next: transforms are ~24.7 of ~41 ms and near lib/ntt's kernel bound; `write()` 6 ms is fixed.
  Five primes are the minimum with 30-bit primes (four give 2^120 < 2^147). Left: the radix-8
  level's reduction (1.2 ms), the subtrees (lib/ntt's kernels).
- 2026-10-10, claude (lib, issue #156 round 2): the huge-page allocation comes from
  `lib/mem/huge.hpp` instead of a local copy. Same stripped executable as before (judge flags,
  `lc-amd`).
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).

- 2026-10-10, claude (round 3). `lc-bench` (EPYC 7B13, 3.49 GHz core clock), judge flags. Hot
  in-process timings unless noted; exploration files in `lc-opt-explore/convolution_mod_2_64/r3`.
  - Transform, 5 primes, lg 20, hot (ms): lib/multimod 24.11; with ntt::Product's bottom stage
    and fused inverse top (problem-local, `product.hpp`) 23.25: tables 0.18, radix-8 level 2.92,
    subtrees 19.06, inverse top 1.06. Each radix-4 level of the subtrees costs ~2.0 ms whether it
    runs from L3, L2 or L1: compute-bound, ~14 cycles per radix-4 butterfly (49 vector ops).
  - Zen 3 throughputs (microbenchmark): `vpmulld` and `vpmuludq` 2 per cycle on the same two
    pipes; `vpsrlq` 2 per cycle on two others; `vpaddd`, `vpminud`, `vpblendd` 4; a 1:1 mix of
    multiplies and adds only 3 per cycle. So ~3 vector ops per cycle is the practical bound here.
  - Radix-8 level on 64-bit input: GCC's loop 62.6 cycles per column (155 vector ops, constants
    spilled). Generated asm (one column per iteration, lib/ntt's scheduler) 53.5: 2.92 -> 2.55 ms
    for the 10 calls. Same output, checked word for word. Lost: two columns per iteration
    57-60 cycles; software pipelining (narrow column j + 1 while column j's butterflies run,
    through two stack buffers swapped each iteration) 54-62 over 42 knob settings.
  - Formatter (in memory, 2^20 values; in the program 2.60 -> 1.95 ms): 2.63 ms -> 1.98. GCC -O2
    kept every `for (h < 2)` loop over halves as a loop, values on the stack, and turned
    `mullo(tens, 2559)` into 4 ops on the shift pipes. Top's 4 characters from a table instead of
    14 vector ops per 4 values. Ablations: all three 1.98; without the kept `vpmullw` 2.04; without
    the unrolling 2.23; table alone 2.32; unrolling and `vpmullw` without the table 2.49; table
    without the existing software pipelining 2.43 (kept). A pitfall on the way: the harness found
    a variant header next to its own source before the `-I` directory, so the first "variants"
    were all one file; the numbers above come from one directory per variant.
  - CRT: GCC kept the loop over the 5 primes in `combine`, loading and splitting each constant per
    step. `#pragma GCC unroll 5`: 1.05 -> 0.88 ms.
  - Phases now (all_same_03, ms, own runner with fork and reap stamps): start 1.05, parse 3.3,
    transforms 23.9, CRT 0.88, format 1.93, `write()` 7.5, exit 1.45; wall 40.1.
  - `judge.py bench`, 21 rounds, slowest 3 cases (ms, median): base 42.60; Product + generated
    radix-8 41.46 (0.9752); + formatter + CRT 40.75 (0.9549). Product alone was 0.9797.
  - convolution_F_2_64 includes `fields64.hpp`: re-bundled, 51/51 official tests, bench 37.83 ->
    36.62 (0.9719).
  - Checks: 44/44 official tests (`lc-amd`); stress 300 rounds (judge flags, lengths to 2^12 + 2:
    lg 9-14 through `product.hpp`, both top levels) and 60 with ASan/UBSan; `test_fields64.cpp`
    4.0M values, native and x86-64-v3; ASan/UBSan build on 9 official cases, file and pipe input.
  - Lost, after #294: variable-width output (a space, the top's digits without padding from a
    table that also holds their count, then mid and low; 20.4 instead of 21 bytes per random
    value; tokens checked on test_fields64's cases and block lengths 1-64). In the program:
    `write()` 7.33 -> 7.23 ms, format 1.95 -> 2.60 (8 field positions per group from a scalar
    chain, stores at computed addresses). Not kept.
  - CI (#294, merged): convolution_mod_2_64 geomean 0.9501 (EPYC 9V45 0.9541, 7763 0.9460,
    0.9501); convolution_F_2_64 0.9705.
  - Submitted the merged `main.cpp` (#294): [409604](https://judge.yosupo.jp/submission/409604)
    AC 41 ms, [409605](https://judge.yosupo.jp/submission/409605) AC 41 ms; `tools/spikes.py`:
    clean 41 ms for both. Previous best 43.
- Next: the CRT as a scheduled asm loop (23 -> ~21 cycles per 8 values, a guess: 0.1 ms). The
  subtrees run lib/ntt's kernels at ~3 vector ops per cycle; `write()` (7.3 ms), start and exit
  (2.5 ms) and parsing (3.3 ms, lib/io) are the rest of the 40 ms.
- 2026-10-10, claude (lib, issue #156 round 3): `product.hpp`, `kernels.hpp` and `gen_kernels.py`
  moved to lib/multimod (`WideProduct`, one kernel set of `multimod::Product`; tests in
  `lib/multimod/test.cpp`). Same instructions but in `Product::multiply` (631 -> 630, other
  registers in the setup); `judge.py bench`, `lc-bench`, 61 rounds: 40.38 -> 40.55 ms (1.0004,
  noise). 44/44 official tests, stress 200 rounds, ASan/UBSan on 7 official cases.
  Details: `lib/multimod/notes.md`.

## Sources

- lib/ntt (our QPoly-derived kernels, ntt::Product, the asm generator) and convolution_mod
  (radix-8 first level, fixed-width output, preinit start). CRT with a floating-point estimate of the multiple of M: a standard
  technique, written here from the formula; no code read.
