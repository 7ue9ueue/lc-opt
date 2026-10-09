# convolution_F_2_64

N, M <= 2^19 elements of F_2[x] / (x^64 + x^4 + x^3 + x + 1) as uint64; print the N + M - 1
coefficients of the product. 10 s.

Best judged: ours, 46 ms: [409244](https://judge.yosupo.jp/submission/409244) (current `main.cpp`,
from #111). Record when the issue opened: 409 ms.

## Design

- Additive FFT in the novel polynomial basis X_j = prod s_i^{j_i} (Lin, Chung, Han, "Novel
  polynomial basis and its application to Reed-Solomon erasure codes", FOCS 2014), on a Cantor
  basis beta_0 = 1, beta_{i+1}^2 + beta_{i+1} = beta_i (computed at compile time). On it
  s_i(beta_i) = 1, stage i's twiddle for the block at c is omega_{c >> i}, and the change from
  monomials to X is XOR-only: Taylor expansion in t = x^(2^K) + x, then rows and columns
  recursively (the Cantor-basis recursion of Gao and Mateer, "Additive fast Fourier transforms
  over finite fields", IEEE Trans. IT 2010; Bernstein and Chou's notes on it).
- One transform length 2^l >= N + M - 1. Basis change of a and b (skipping their zero upper
  halves), forward transform of a, then one depth-first pass that transforms b, multiplies it into
  a and transforms a back block by block (32 KiB blocks in L1), then the basis change back.
- Field multiply, 4 lanes: two `vpclmulqdq` (lanes 0/2 and 1/3), unpack, reduction by shifts plus
  a `vpshufb` table for the top nibble. Fallbacks: xmm `pclmulqdq`, then portable (both tested).
- Basis change: the top Taylor expansion (16 rows of 2^16) runs in one pass as a Taylor shift by
  x with one carry, on 128-column blocks right to left; rows of 2^K <= 256 single words go four at
  a time through a 4x4 transpose; column steps run on column blocks of at most 64 KiB.
- `lib/io` input (`read_bulk`); `.preinit_array` start, `_exit`.
- Output (`fields.hpp`, judge-specific): blocks of 3104 values. A block where at least 1/8 of
  the values are below 2^53 goes through `write_array`; otherwise every value is right-aligned in
  20 characters (wcmp checker compares tokens). Scalar code splits x = h 10^16 + m 10^8 + l and
  looks up h as 4 characters with leading spaces; AVX2 makes 8 digits of m and l, four values
  per step, 64 values behind the scalar split in the same loop. h = 0 (rare) is blanked after.

## Costs (lc-amd, gen_max, ms, in-process stamps)

read 2.4, basis change of a 2.1, of b 2.1, forward a 5.8, fused b forward + product + inverse
11.8, basis change back 4.3, format 4.9, `write()` to tmpfs 7.4. Whole process ~44.

- Zen 3 `vpclmulqdq` ymm: 4.5 TSC ticks per pair (one multiply of 4 lanes), the bottleneck;
  the transforms do ~31.5M 64-bit products, ~14.5 ms at that rate, so they run at ~80% of it.
  A reduction by two more clmul folds is 2.3x slower (10.5 ticks).
- `write()` of 21 MB into tmpfs costs 7.4 ms; `lib/io/notes.md` found it irreducible.

## Log

- 2026-10-09, claude (round 1): first solution as above, built in steps. `tools/judge.py test`
  51/51 on lc-amd; stress 300 rounds vs `brute.cpp` for the VPCLMULQDQ, PCLMUL, portable and
  ASan/UBSan builds (scratch script, sizes up to ~2^11); ASan/UBSan and both fallbacks exact on
  the large official cases. Slowest case on lc-amd per step:
  - v1, stage-by-stage transforms, level-by-level basis change: 59.2 ms. Phases: basis change
    5.5 per input and 6.4 back, transforms 7.2 each.
  - Zero upper halves skipped, column blocks, radix-4 passes above 32 KiB, b's forward fused with
    the product and a's inverse, `vpshufb` reduction: 47.9 ms.
  - Top Taylor in one pass, transposed short rows: 45.5; vector shift in that pass: 44.0.
  - Paired stages inside the 32 KiB blocks: neutral (bench ratio 0.999, 15 rounds), kept.
  - `judge.py bench` v1 vs final, 11 rounds: 60.11 vs 47.03 ms median, ratio 0.78.
- 2026-10-09, claude: submitted #111's `main.cpp`: [409244](https://judge.yosupo.jp/submission/409244),
  AC 46 ms, 37.4 MiB (1/5 for this version).
- 2026-10-09, claude (round 2): fixed-width u64 output, `fields.hpp`. lc-amd, judge flags.
  Format of 2^20 random values into memory, median of 15 (scratch harness):
  - Scalar split into u32 arrays, then AVX2 digits (64-bit lanes, three `x + q (2^k - d)` steps
    and a byte reverse): 3.45 ms (split 1.55, digits and stores 1.67). Digits with
    quotient/remainder/shift/or per step instead: 2.21 for the digit pass.
  - Split in AVX2 with double-precision quotients (`fma`, `cvttpd2dq`, one integer correction),
    h's digits in AVX2 too: 5.42; next group's split issued before current digits: 4.13; split
    as its own pass: 3.75. Vector integer multiplies share one port with the digit steps.
  - Scalar split into GPRs, `_mm256_setr_epi64x` into the digit code: 4.03.
  - Scalar split 64 values ahead in the digit loop (kept): 3.15 (16 ahead 3.15, 256 ahead 3.19).
  - Old `write_array`: 4.9 ms (round 1, in-process stamp).
  - Fixed width on every block: `judge.py bench` ratio 1.21 on all_ones/all_same/small_values
    (their output is short; padding costs more in `write()`). Blocks with >= 1/8 short values
    now stay variable-width.
  - Kept version vs round 1, `judge.py bench`, 15 rounds, 6 slowest cases: 45.62 vs 44.31 ms
    median, ratio 0.959. `judge.py test` 51/51, slowest 43.0 ms. Stress 400 rounds vs `brute.cpp`
    (judge flags, ASan/UBSan, x86-64-v3); ASan exact on gen_max, small_values, all_same, all_ones.
  - CI bench vs main: ratio 0.958 (Xeon 6973P-C), 0.972 and 0.962 (EPYC 7763). Merged as #119.
- 2026-10-09, claude: submitted #119's `main.cpp` twice: [409257](https://judge.yosupo.jp/submission/409257)
  AC 53 ms (many_ones_00 53, next 44) and [409259](https://judge.yosupo.jp/submission/409259) AC 50 ms
  (random_01 50, next 45). Each has one spiked case; the rest top out at 44-45 vs 46 in 409244.
  Best judged stays 46 ms (2/5 for this version).
- Next: the basis change (~8.5 ms, ~20 XORs per element at the store limit would be ~2 ms per
  change); `write()` (7.4 ms) and the transforms (~80% of the PCLMUL limit) are near their floors.
