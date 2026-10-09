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
- `lib/io` input (`read_bulk`), `write_array` output; `.preinit_array` start, `_exit`.

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
- Next: u64 formatting with SIMD (4.9 ms now); the basis change (~8.5 ms, ~20 XORs per element
  at the store limit would be ~2 ms per change).
