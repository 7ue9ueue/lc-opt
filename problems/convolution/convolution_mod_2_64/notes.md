# convolution_mod_2_64

N, M <= 2^19 coefficients below 2^64; print the N + M - 1 coefficients of the product mod 2^64.
10 s. Record when opened: 76 ms (another user). Best judged: ours, 45 ms:
[409250](https://judge.yosupo.jp/submission/409250) (current `main.cpp`).

## Design

- Five NTT primes below 2^30 with 2^20 | p - 1: 998244353, 985661441, 976224257, 975175681,
  972029953. Product 2^149.35 > 2^19 (2^64 - 1)^2. One cyclic transform of length 2^lg per prime.
- `transform.hpp`: lib/ntt's transform (recursion, tables, kernels) with the modulus a run-time
  value. `kernels.hpp` is lib/ntt's generated kernels with P, 2P, -1/P as variables instead of
  constants (`gen_kernels.py` imports lib's generator and rewrites 6 lines; same instructions).
  For 2^lg = 2 * 4^j >= 256 with both factors at most half (all large tests) the first level is a
  radix-8 pass that reads the 64-bit input and reduces it mod p on the fly
  (hi (2^32 mod p) + lo; lo < 8p brought below 2p by two halvings).
- CRT, not Garner: the transform for prime k returns y_k = c / M_k mod p_k (the factor is folded
  into the final scale). c = sum y_k M_k - t M with t = floor(sum y_k / p_k); the fraction is
  c / M < 0.2, so t comes from a float sum + 0.4. Then c mod 2^64 from 64-bit constants
  C = lo + 2^32 hi: y lo by `vpmuludq` (even and odd lanes apart), y hi mod 2^32 by `vpmulld` in
  8 lanes. 18 multiplies per 8 values, no dependency chain between primes.
- Memory: a, b (64-bit), work, residues of primes 0-3: 7 arrays of 4 MiB at the largest size.
  The last prime writes its residues to work and uses a's storage as its work.
- Output (`fields64.hpp`, judge-specific): every value right-aligned in 20 characters after a
  space. top = x / 10^16, mid, low from multiply-shift estimates on the high bits, corrected once.
  mid and low -> 4-digit chunks -> digit bytes in 16-bit lanes; two stores per value (8 bytes of
  top, 16 of mid+low). Separators are never written: the buffer starts as spaces. Leading-zero
  logic for mid and low runs only when some top in the 8 is 0.
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
- Next: transforms are ~24.7 of ~41 ms and near lib/ntt's kernel bound; `write()` 6 ms is fixed.
  Five primes are the minimum with 30-bit primes (four give 2^120 < 2^147). Left: the radix-8
  level's reduction (1.2 ms), the subtrees (lib/ntt's kernels).

## Sources

- lib/ntt (our QPoly-derived kernels) and convolution_mod (radix-8 first level, fixed-width
  output, preinit start). CRT with a floating-point estimate of the multiple of M: a standard
  technique, written here from the formula; no code read.
