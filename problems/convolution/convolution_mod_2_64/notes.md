# convolution_mod_2_64

N, M <= 2^19 coefficients below 2^64; print the N + M - 1 coefficients of the product mod 2^64.
10 s. Record when opened: 76 ms (another user). Best judged: none yet.

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
  c / M < 0.2, so t comes from a float sum + 0.4. Then c mod 2^64 from 64-bit constants: 12
  `vpmuludq` per 4 values, no dependency chain between primes.
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
- Next: the transforms are 25 of 43 ms (kernel-bound, as convolution_mod); `write()` 6 ms is fixed.
  Ideas: CRT fused into the last prime's final pass (its residues are hot); fewer page faults
  (residues 5 x 4 MiB fresh); parse into the reduced form directly.

## Sources

- lib/ntt (our QPoly-derived kernels) and convolution_mod (radix-8 first level, fixed-width
  output, preinit start). CRT with a floating-point estimate of the multiple of M: a standard
  technique, written here from the formula; no code read.
