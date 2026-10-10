# convolution_F_2_64

N, M <= 2^19 elements of F_2[x] / (x^64 + x^4 + x^3 + x + 1) as uint64; print the N + M - 1
coefficients of the product. 10 s.

Best judged: ours, 42 ms: [409339](https://judge.yosupo.jp/submission/409339) (current `main.cpp`,
#167). Earlier: 44 ms, [409295](https://judge.yosupo.jp/submission/409295) (#119); 46 ms,
[409244](https://judge.yosupo.jp/submission/409244) (#111). Record when the issue opened: 409 ms.

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
- Field multiply-add, 4 lanes: two `vpclmulqdq` (lanes 0/2 and 1/3), unpack, reduction by shifts
  (`high << 1` as `vpaddq`) plus a `vpshufb` table for the top nibble; the last shift's term is
  added last (an empty asm stops GCC reassociating it). Fallbacks: xmm `pclmulqdq`, then portable
  (both tested).
- Transform passes: three stages per pass (eight vectors, four independent products per stage)
  above stage 5, then stages 5-4, then two passes over 16-word groups: stages 3-2, and stages 1-0
  after a 4x4 transpose. Group twiddles step in registers (one XOR with a table entry chosen by
  ctz(g + 1)). The forward transform leaves each 16-word group transposed; the inverse starts
  there.
- Basis change: change_basis<4> is a register kernel whose XOR list is recorded at compile time by
  running the same recursion on element indices. L = 20: the column step (change_basis<4> across
  16 rows) runs inside the one-pass top Taylor step (Taylor shift by x, 128-column blocks, right
  to left). L = 16 (rows of 512 KiB, the size of L2) in three passes: the top four Taylor levels
  in sheared columns (g_r[J] = f_r[J - r] turns the shift into a 16-element superset sum per
  column, in place), then per 32 KiB block the bottom four levels and the row step (four rows at
  a time through a 4x4 transpose), then the column step on slices of two lines per row gathered
  into contiguous buffers.
- `lib/io` input (`read_bulk`); `.preinit_array` start, `_exit`.
- Output (`fields.hpp`, judge-specific): blocks of 12288 values. A block where at least 1/8 of
  the values are below 2^53 goes through `write_array`; otherwise
  `../convolution_mod_2_64/fields64.hpp` prints a space and each value right-aligned in 20
  characters (wcmp checker compares tokens). Its text is in b (dead by then), and each block goes
  to `write(2)` directly. Until round 5 the fixed-width blocks had their own formatter (round 2
  in the log).

## Costs (lc-amd, gen_max, ms, in-process stamps)

Round 3 (TSC at 3.05 GHz; core about 3.49 GHz, TSC = 0.874 core cycles): read 2.55, basis change
of a 1.25, of b 1.15, forward a 5.24, fused b forward + product + inverse 10.6, basis change back
2.5, format and `write()` to tmpfs 11.2. Whole process ~38 (`judge.py`). Main before round 3,
same harness: 2.55, 2.1, 2.05, 6.0, 12.1, 4.3, 11.2.

- Zen 3, core cycles (`ub_mix`, round 3): `vpclmulqdq` ymm alone 2.0 each, so 4.0 per vector
  multiply. The multiply's whole instruction mix, independent: 4.91; + load, store and two
  butterfly XORs: 5.07. Shuffle-port ops (unpack, shift, `vpshufb`) beyond the four that the two
  clmuls leave free cost ~0.45 each (without srl60 + `vpshufb`: 4.00). Loads, stores and XORs are
  nearly free.
- Multiply + XOR latency 12.8 cycles (13.7 before the late-shift order). Chains in registers: 1:
  12.8, 2: 7.7, 3: 6.2, 4: 5.8, 8: 5.7 cycles per multiply. The transform loops run at one
  iteration per iteration-latency (little overlap): radix-4 6.6-6.9, radix-8 6.35-6.45, group
  passes 6.8 (stages 3-2) and 7.8 (1-0), in-place multiply 5.5.
- Transforms: ~31.5M products; at 5.8 cycles per 4 they would take ~13 ms, now 15.8.
- A reduction by two more clmul folds is 2.3x slower (round 1).
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
- 2026-10-09, audit (claude): resubmitted #119's `main.cpp`, [409295](https://judge.yosupo.jp/submission/409295)
  AC 44 ms (3/5), no spike: large cases 40-44 ms against 44.3 expected (`judge.py bench`).
- 2026-10-09, claude (round 3, assembly pass). Survey of candidates first (from each problem's
  notes): gcd/lcm convolution (compute ~4.5 ms over an 11.3 ms floor, memory-bound sweeps),
  mul_mod2n (~8 ms compute, half in lib/ntt asm kernels), min_plus concave (branchy, load-latency
  bound), bitwise_xor (radix-16 already generated). Picked this one: ~26 of 44 ms compute, all
  compiler-generated. lc-amd, judge flags; harness timings are medians of 15-31 runs on 2^20
  random words (scratch harnesses `hb_cb`, `hb_stage`, `sp_test`, `ub_mix`).
  - Pieces of the inverse basis change before: 4.20 = columns<20,16> 0.39 + 16 x change_basis<16>
    3.24 (column step 1.67, row step 1.01, taylor<2^16,256> 0.61) + top Taylor 0.57. An 8 MiB copy
    takes 0.2 ms: the cost is in-cache work, not DRAM.
  - change_basis<4> as a register kernel (16 loads, 44 XORs, 16 stores): row step 1.01 -> 0.54;
    on 512 KiB strides worse, 0.39 -> 1.07 (16 rows share one L1 set and one L2 set).
  - Column slices of one line gathered into a buffer: column step 1.67 -> 0.86 (gather and
    scatter alone 0.55). Software prefetch (T0, T1, next 1-2 slices) and `prefetchw` before the
    scatter: no gain or worse. Two lines per row visit into two buffers: 0.82 (kept); one 16-word
    buffer 0.90, four lines 0.86. Gather alone 0.24, scatter alone 0.35.
  - L = 20 column step inside the top Taylor pass: inverse 3.16 -> 2.62.
  - Top four levels of taylor<2^16,256> as one sheared pass: 0.39 -> 0.22; three-pass
    change_basis<16>: inverse 2.59 -> 2.46.
  - `high << 1` as `vpaddq`: forward transform 5.93 -> 5.77, fused pass 12.00 -> 11.70.
  - Stages 2-0: the 8-word kernel was one dependency chain (12.1 cycles per vector multiply,
    1.70 ms per transform). 32-word kernel after 4x4 transposes: 1.07; restructured to cut
    spills: 1.14; output left transposed: 1.05. 16-word kernel for stages 3-0: 1.46 (chain four
    multiplies deep). Pairs down to stage 2 plus a kernel for stages 1-0: forward 5.40, fused 10.93.
  - Group passes for stages 3-2 and 1-0 with twiddles stepped in registers instead of scalar
    lookups and GPR broadcasts: forward 5.27, fused 10.65.
  - Radix-4 loop with two columns interleaved: 8.04 cycles per multiply (spills; worse).
    Software-pipelined radix-4 (level 2 of column j with level 1 of j + 4): 6.31 vs 6.62 on long
    loops, 7.09 vs 6.88 on 4-iteration calls. Not kept.
  - Late shift term in the reduction (empty asm): latency 13.7 -> 12.8; forward 5.21, fused 10.56.
  - Radix-8 passes: 6.35 vs 6.62 cycles per multiply; forward 5.14, fused 10.49.
  - perf on lc-intel (x86-64-v3 + extensions, gen_max x10): triple_forward 15.8%, format 9.8%,
    triple_inverse 7.2%, input parser 6.2%, kernel ~12%, group passes 4%, column steps 2.9%.
  - Result: `judge.py bench` vs main, 21 rounds, 6 slowest cases: 43.79 -> 38.12 ms, ratio 0.867.
    `judge.py test` 51/51 (slowest 37.2). Stress 400 rounds vs `brute.cpp` (judge flags,
    ASan/UBSan, x86-64-v3 + PCLMUL, portable); 41 random cases at every length 2^4..2^20 against
    main's binary for the judge, PCLMUL and portable builds; ASan exact on gen_max, small_values,
    all_same, all_ones, many_ones.
  - CI: ratio 0.8675 (EPYC 7763 x3: 0.873, 0.865, 0.864; medians 44.52 -> 38.88, 47.51 -> 41.66,
    47.52 -> 41.39 ms). Merged as #167.
- 2026-10-10, claude: submitted #167's `main.cpp` twice. [409339](https://judge.yosupo.jp/submission/409339)
  AC 42 ms: gen_2_x_3_11_01 42 (32.6 on lc-amd, a spike), every other case at most 38.
  [409345](https://judge.yosupo.jp/submission/409345) AC 46 ms: many_ones_00 46, every other case at
  most 38. Large cases 34-38 ms against 43-44 for #119. Best judged 42 ms (2/5 for this version).
- 2026-10-10, claude (lib/io #21, round 5): fixed-width blocks through
  `../convolution_mod_2_64/fields64.hpp`. In memory, 2^20 random values, median of 31 (`lc-amd`,
  judge flags): fields64 2.62 ms, this problem's round-2 formatter 3.19. Blocks of 12288 values
  (was 3104) go to `write(2)` directly, from b (page-aligned, at least 2^15 words for every
  size, so no static buffer), instead of 65516-byte Writer flushes whose start drifted against
  the page (0.6% of bytes in the slow band, `lib/io/notes.md`). Short-value blocks are
  unchanged but now start with a space. Per case, 15 rounds: gen_max_00 37.85 → 36.78 ms,
  many_ones_00 38.09 → 37.12, random_00 33.95 → 33.07; all_ones, all_same, small_values
  (variable width) equal. `judge.py bench`, 31 rounds, 6 slowest cases: 39.62 → 39.23 ms
  (0.976). Checks: `judge.py test` 51/51; 120 random inputs (runs of short and long values
  across block edges with a = 1, long outputs, tiny ones) token-identical to main, also with
  ASan/UBSan, file and pipe input.
  PR #198 merged; CI 0.9922 (EPYC 7763 0.969; EPYC 9V45 0.998 and 1.011). Submitted
  [409429](https://judge.yosupo.jp/submission/409429): AC 45 ms, 38.0 MiB. Spikes: all_ones_00
  45 (36 in 409339 and 409345), random_01 44 (35, 34), gen_2_x_3_11_00 41 (33, 32); the rest at
  most 38 (many_ones_00 38, gen_max 37). Clean 38, as before; best judged stays 42.
- Next: the transform loops run at ~6.4 cycles per vector multiply against 5.7 with eight
  independent chains: a hand-scheduled asm loop with two columns in flight and no spills
  (~1.5 ms if it reaches 5.8). Fuse the pointwise product into the stage 1-0 group passes (shared
  twiddles, no store and reload). Column gather and scatter still ~0.4 ms per inverse change;
  top Taylor and bottom four levels of taylor<2^16> as sheared passes with unaligned loads.
