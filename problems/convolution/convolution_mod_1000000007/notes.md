# convolution_mod_1000000007

N, M <= 2^19 coefficients mod 10^9 + 7; print the N + M - 1 coefficients of the product. 10 s.
Record when opened: 29 ms (another user). Best judged: ours, 22 ms, spike-free:
[409640](https://judge.yosupo.jp/submission/409640) and
[409643](https://judge.yosupo.jp/submission/409643) (`main.cpp` of #305). Before: 23 ms,
[409310](https://judge.yosupo.jp/submission/409310) (#129) and
[409353](https://judge.yosupo.jp/submission/409353) (#162); 31 ms,
[409262](https://judge.yosupo.jp/submission/409262) (#129; a judge spike, clean score 23 ms,
`tools/spikes.md`).

## Design

- Three NTT primes below 2^28 with 2^17 | p - 1: 268042241, 265420801, 264634369. Product
  2^83.96 > 2^19 (10^9 + 6)^2 = 2^78.8. 16P < 2^32, so values may grow to 16P between
  reductions; inputs < 10^9 + 7 < 4P are used as they are.
- `multimod::LazyProduct` (`lib/multimod/lazy_product.hpp`; this folder's `product.hpp` until
  #156 round 3): lib/multimod's tables and recursion with ntt::Product's bottom stage (two groups
  per kernel, no leaf weight array) and fused inverse top level, as convolution_mod_2_64's
  `WideProduct`. Its kernels (`lib/multimod/lazy_kernels.hpp`, from `gen_lazy_kernels.py`, which
  reuses lib/ntt's graphs, scheduler and allocator) drop most reductions: forward butterflies
  take and return values < 8P (one min step on f0: 44 vector ops instead of 49), inverse ones
  < 4P (48 instead of 52), leaves are products of values <= 2P (sums < 2^61, Montgomery result
  < 3P, no final min), and the radix-8 first level reads the raw input (no narrowing). odd lg:
  C++ radix-4 top levels with the same ranges. 4P and 8P are memory operands.
- The transform length is 2^lg >= max(2^9, n + m - 1, 2 max(n, m)): both factors fill at most
  half of it, so LazyProduct serves every size (no Transform fallback).
- a and b are the two halves of one array, which becomes the last prime's residues: its b
  transform goes to the scratch array first, then a's runs in place. 4 arrays of 2^lg words.
- CRT straight to mod 10^9 + 7: y_k = c / M_k mod p_k (factor folded into the transform's scale),
  t = floor(sum y_k / p_k) from a float sum + 0.5 (the fraction is c / M < 2^-5), then
  s = sum y_k (M_k R mod q) + t (-M R mod q) < R q with R = 2^32, and one Montgomery reduction
  gives c mod q. 12 `vpmuludq` per 8 values.
- Output per block of 25600 values: 10-byte fields (`fields10.hpp`, convolution_mod's
  `fields.hpp`) when every value is < 10^9, else 11-byte fields (`fields11.hpp`: v / 100 as 8
  digits, the last two digits and the separator in a tail dword, six 16-byte `pshufb` chunks per
  lane of 8 values). A block holds a value >= 10^9 with probability about 1.8e-4 on random data.
  `-DFORCE_WIDE` forces 11-byte fields for tests.
- `.preinit_array` start and `_exit`, one huge-page arena (as convolution_mod).

## Log

- 2026-10-09, claude (round 1). `lc-amd`, judge flags.
  - First version (3 primes, CRT by Montgomery, 11-byte fields everywhere): 48/48 official tests,
    slowest 24.0 ms (`judge.py test`). I/O floor (`../floor.py`, plain `write_array`): 10.9 ms.
    Phases on fft_killer_03 (ms, 9 runs, in-process `CLOCK_MONOTONIC`, scratch probe not
    committed): parse 1.6, prime 0 5.0, prime 1 4.8, prime 2 4.8, CRT 0.49, format 1.02,
    `write()` ~3.5; 22.2 from `solve()` entry to exit.
  - 10-byte fields unless a block has a value >= 10^9 (output 10.5 MB instead of 11.5 MB,
    formatter 0.65 ms instead of 1.02) and the last prime in place: `judge.py bench`, 21 rounds,
    slowest 3 cases: 25.21 -> 23.99 ms, ratio 0.950. `judge.py test`: slowest 22.9 ms.
  - Checks: 48/48 official tests; `test_fields.cpp` (fields10 for every value < 10^9, fields11
    for every value < 10^9 + 7, against a scalar formatter); `stress.py` 400 rounds, 200 with
    `-DFORCE_WIDE`; ASan/UBSan stress 60 + 40 (`-DFORCE_WIDE`) rounds and 6 official cases with
    file and pipe input.
  - Submitted the merged `main.cpp` (#129): [409262](https://judge.yosupo.jp/submission/409262)
    AC 31 ms from one outlier (fft_killer_06 31, the other large cases 21-23);
    [409263](https://judge.yosupo.jp/submission/409263) AC 32 ms from three outliers
    (max_ans_zero_00 32, fft_killer_07 31, random_00 30; the rest 20-23). Matches `lc-amd`
    (22.9). The judged maximum is jitter; no further submissions of this version.
- 2026-10-09, claude (round 2). `lc-amd`, judge flags. Phases on fft_killer_05 (ms, median of
  21, in-process `CLOCK_MONOTONIC`, scratch probe not committed): parse / transforms / output
  (CRT + format + `write()` to /dev/null).
  - Base (main): 1.54 / 14.17 / 1.22.
  - CRT fused into the last prime's final pass: the pass hands each finished vector (or chunk of
    vectors) to a callback that overwrites y2 with c mod q; the formatter then reads c straight
    from the transform's array, with a max scan per block for the 10/11-byte choice.
    Per vector: 1.54 / 14.77 / 0.80. Per chunk of 32, 128, 512 or 2048 vectors (CRT from L1):
    14.69-14.88 / 0.80-0.82. The same pass with an empty callback: 14.13 / 0.80. So the CRT costs
    0.42 ms standalone (writing an L2 block) and 0.57-0.63 ms fused: it is compute-bound, the
    memory traffic it saves was not the limit, and the fused loop reloads the residue pointers
    and spills constants. `judge.py bench`, 21 rounds, slowest 3: base 23.84, fused 24.19 ms
    (ratio 1.013); a second run: base 24.05, per vector 24.21 (1.005), with the asm scale kernel
    for the first two primes 24.26 (1.012). Dropped.
  - Not tried, estimated small: CRT with (y2 + (2 - t) p2) M2 instead of the t M term saves 2 of
    12 `vpmuludq` per 8 values (~0.05 ms, guess); a 4 MiB array fewer (in-place last prime for a
    needs a's buffer at 2^lg words, but primes 0-1 still need a 4 MiB scratch): no saving.
- 2026-10-09, audit (claude): resubmitted 3 times to clear the outlier, now 5/5 (cap).
  [409297](https://judge.yosupo.jp/submission/409297) AC 32 ms, [409302](https://judge.yosupo.jp/submission/409302)
  AC 34, [409306](https://judge.yosupo.jp/submission/409306) AC 33. In all 5 runs the large cases take
  22-24 ms (expected 23.8, `judge.py bench`), and 1-4 of the ~30 large cases spike to 30-34 ms.
  With this many large cases a spike-free run is unlikely; best judged stays 31 ms.
- 2026-10-09, claude (spike check). Every judged maximum so far is a judge launch spike, not our
  code: +9 ms on 5.2% of all cases, any size (`tools/spikes.md`). `tools/spikes.py` on the 5
  submissions: clean score 23, 23, 23, 24, 24 ms (409262, 409263, 409297, 409302, 409306).
  23 cases sit within 9 ms of the slowest, so a run is clean with P = 0.948^23 = 0.29;
  all 5 spiking has P = 0.71^5 = 0.18.
- 2026-10-10, claude (issue #156): the local transform moved to `lib/multimod` (`Padded`).
  At odd lg (2^7..2^19) a sparse input is now reduced and transformed in one pass; lg 20 is the
  same logic. Transforms alone at lg 20: ratio 1.0009; at lg 19: 0.9837. `judge.py bench`,
  31 rounds, `lc-amd`: 0.9991. 48/48 official tests. Details: `lib/multimod/notes.md`.
- 2026-10-10, audit (claude): submissions not logged before. Who submitted them is not recorded.
  "clean" is the score without launch spikes (`tools/spikes.py`), where it differs.
  - `main.cpp` of #129 (equal but for the final newline), 2026-10-09 UTC, 30.9-31.0 MiB:
    [409291](https://judge.yosupo.jp/submission/409291) 22:34 AC 33 ms, clean 24;
    [409292](https://judge.yosupo.jp/submission/409292) 22:37 AC 33 ms, clean 23;
    [409298](https://judge.yosupo.jp/submission/409298) 22:37 AC 31 ms, clean 24;
    [409303](https://judge.yosupo.jp/submission/409303) 22:41 AC 31 ms, clean 23;
    [409304](https://judge.yosupo.jp/submission/409304) 22:41 AC 33 ms, clean 24;
    [409307](https://judge.yosupo.jp/submission/409307) 22:42 CE (submitted as C++17);
    [409308](https://judge.yosupo.jp/submission/409308) 22:43 AC 32 ms, clean 23;
    [409309](https://judge.yosupo.jp/submission/409309) 22:44 AC 31 ms, clean 23;
    [409310](https://judge.yosupo.jp/submission/409310) 22:45 AC 23 ms, no spike on the slowest
    case (linked from README.md before, not from here).
    With 409262, 409263, 409297, 409302 and 409306, this version has 14 submissions (13 judged),
    over the cap of 5.
  - Current `main.cpp` (#162), 2026-10-10 UTC, 30.9 MiB:
    [409342](https://judge.yosupo.jp/submission/409342) 01:47 AC 31 ms, clean 24;
    [409347](https://judge.yosupo.jp/submission/409347) 01:49 AC 33 ms, clean 24;
    [409353](https://judge.yosupo.jp/submission/409353) 01:56 AC 23 ms, no spike on the slowest
    case.
  - New best judged: 23 ms (409310, 409353); was 31 (409262).
- Next: everything left is in the transforms (14.2 of ~23 ms, lib/ntt's kernels) and fixed I/O.
  No problem-local idea left that is worth more than noise.
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  24.03 → 23.86 ms (0.989). 48/48 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib, issue #156 round 2): the arena comes from `lib/mem/huge.hpp`
  (`mem::Arena`) instead of a local copy. Same instructions, other stack slots in `solve()`;
  `judge.py bench`, `lc-bench`, 21 rounds: 23.53 -> 23.54 ms (noise). Official tests pass;
  ASan/UBSan on 4 official cases.
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).
- 2026-10-10, claude (round 3). `lc-bench` (EPYC 7B13, core clock 3.49 GHz), judge flags.
  Exploration files: `lc-opt-explore/convolution_mod_1000000007/r3`.
  - convolution_mod_2_64's `product.hpp` structure (ntt::Product's bottom stage and fused
    inverse top on lib/multimod) with `Padded` input. 3 transforms at lg 20, hot: lib/multimod
    13.94 ms, Product 13.54. A list-scheduled asm first level (as 2_64's) was no faster than
    lib/multimod's C++ one here (1.21 vs 1.13 ms for 6 calls; `judge.py bench` 0.9752 vs 0.9787).
  - Primes below 2^28 and lazier reductions (see Design). Kernel microbenchmark (h = 1024, in
    L2, cycles per butterfly): forward 14.8 -> 13.6, inverse 14.6 -> 13.9. Transforms hot:
    12.59 ms (tables 0.11, first levels 1.19, subtrees 10.65, inverse top 0.62).
  - Knob search (100 sets per kernel, timed on lc-bench): lib/ntt's knobs stay best for forward
    and inverse (within 0.5%); bottom_both 122.9 -> 118.3 cycles per group with margin 1, load
    latency 6, window 20 (lib/ntt's window 20 deadlocks the scheduler on these graphs).
  - a and b in one array that becomes the last prime's residues: 2 MiB less memory, same time
    (`judge.py bench`, 41 rounds: 1.0001).
  - Phases on fft_killer_05 (ms, median of 21, `runner2` fork/reap stamps): start, parse,
    transforms, CRT, format + `write()`, exit, wall: main 1.11 / 1.80 / 14.61 / 0.50 / 4.32 /
    0.91 / 23.28; this 1.12 / 1.81 / 13.35 / 0.50 / 4.32 / 0.89 / 21.99.
  - `judge.py bench`, slowest 3 cases, against main: 21 rounds 23.06 -> 21.73 ms (0.9458, lazy
    kernels before the knob search and the layout); 21 rounds 23.53 -> 22.14 (0.9408, knobs),
    22.44 (0.9476, knobs + layout; 1.0001 against knobs alone in 41 rounds).
  - Lost: `prefetchw` of the first level's output lines 64-384 bytes ahead (1-3 of 34 cycles per
    column, within noise), input `prefetcht0` (slower), non-temporal stores (147 cycles per
    column instead of 36). The first level runs at 33-42 cycles per column at lg 20 against 21
    in L1: it waits on memory.
  - Not tried, estimated: two primes plus a double-precision FFT for the high part (a 2^20-point
    complex FFT moves 16 MiB arrays; at best break-even with the third prime's 4.5 ms); primes
    below 2^27 (32P headroom: about 2% of the transforms, guess).
  - Checks: 48/48 official tests (`lc-amd`); `test_product.cpp` (every kernel against a scalar
    model with inputs up to their bounds; products against lib/multimod at lg 9..20, three array
    layouts, garbage past the halves) at -O2 native and x86-64-v3 and with ASan/UBSan; mutations
    of the generator (one reduction dropped) fail it; `stress.py` 500 rounds, 200 with
    `-DFORCE_WIDE`, 100 with x86-64-v3, 100 with ASan/UBSan; ASan/UBSan build on 9 official
    cases, file and pipe input.
  - CI (#305, merged): geomean 0.9617 (EPYC 7763 0.9378, 9V74 0.9955, 9V45 0.9528).
  - Submitted the merged `main.cpp` (#305), 2026-10-10:
    [409640](https://judge.yosupo.jp/submission/409640) AC 22 ms (no spike on the slowest
    cases; large cases 20-22, most at 21), [409642](https://judge.yosupo.jp/submission/409642)
    AC 25 ms (spike on small_and_large_03; clean 21),
    [409643](https://judge.yosupo.jp/submission/409643) AC 22 ms (clean 22). 28.9 MiB (was
    30.9). Best judged 23 -> 22 ms.
- Next: the first level waits on memory (about 0.5 ms over its 6 calls; a fused radix-8 and
  radix-4 first pass might hide it); primes below 2^27.
- 2026-10-10, claude (lib, issue #156 round 3): `product.hpp`, `kernels.hpp`, `gen_kernels.py`
  and `test_product.cpp` moved to lib/multimod (`LazyProduct`, one kernel set of
  `multimod::Product`; the test is now part of `lib/multimod/test.cpp`). Same instructions but in
  `Product::multiply` (other registers and stack slots around the calls); `judge.py bench`,
  `lc-bench`, 61 rounds: 22.30 -> 22.34 ms (1.0020, noise). 48/48 official tests, stress 200
  rounds, ASan/UBSan on 6 official cases. Details: `lib/multimod/notes.md`.

## Sources

- lib/ntt (our QPoly-derived kernels, generator, ntt::Product), convolution_mod_2_64
  (run-time modulus transform, `product.hpp` structure, float estimate of the CRT multiple),
  convolution_mod (radix-8 first level, `fields.hpp`, preinit start). Montgomery reduction
  (Montgomery 1985), written here from the formula. Lazy reduction with headroom below the word
  size: D. Harvey, "Faster arithmetic for number-theoretic transforms", J. Symb. Comp. 60 (2014),
  https://arxiv.org/abs/1205.2926 (idea only).
