# bitwise_xor_convolution

N <= 20; a, b of 2^N values < 998244353; print c_k = sum over i xor j = k of a_i b_j mod P. 5 s.
Input ~20.7 MB (2^21 tokens), output ~10.4 MB.

Best judged: ours, [409210](https://judge.yosupo.jp/submission/409210), 14 ms (round 2).
Round 1: 409198, 15 ms.
Record when opened: 25 ms.

## Design

- c = H (H a . H b) / 2^N, H the Walsh-Hadamard transform: 3 transforms of 20 levels.
- Signed 64-bit lanes, no reduction inside a transform: |x| <= 2^20 P < 2^50.
  Zen 3 runs `vpaddq`/`vpsubq` 4 per cycle; lazily reduced 32-bit lanes need ~3.5 ops per vector
  per level (0.44 per value) against 1 op per 4 values here.
- Rows of 2^12 values (32 KiB of int64), 2^(N-12) rows, rows padded by 64 bytes so a column's rows
  hit different cache sets. Row levels in L1; column levels per strip of one cache line per row.
- The two lane bits: a 4x4 transpose between two radix-4 steps (forward leaves groups of four
  transposed; the product does not care; the inverse transposes back).
- Radix-16 passes in inline assembly (`gen_radix16.py`, which simulates its output): 64 add/sub,
  17 stores, one spill, scheduled so that 16 registers are live only at the end.
- x mod P for |x| < 2^51: the bits of x + 0x4338... are the double 1.5 2^52 + x; one FMA with
  1/P and the offset M - M/P rounds the quotient into the low mantissa (error < 0.9, so |r| < 0.9 P).
- a / 2^N mod P (round 2): Montgomery with R = 2^N. k = -a / P mod 2^N (one `vpmuludq`, an
  and), (a + k P) / 2^N lies in (-P, 2P); its bits N to N + 31 are the signed dword.
- Products in doubles (round 2): h = x y, l = fma(x, y, -h) exact, q = round(h / P) by the magic
  constant (|h / P| < 2^51), r = h - q P exact by FMA; r + l, |r + l| < 0.9 P. No reduction of
  y (|y| < 2^20 P) first; 4 FMA-pipe and 4 add-pipe ops per vector, against 7 multiplies before.
- Order: a's rows while it is parsed (chunks of 2^16 tokens: smaller bulk reads leave ~1000
  tokens per call to the scalar path), a's columns, a / 2^N kept as dwords (4 MiB); b reuses the
  int64 array: rows, columns with the products, inverse columns; inverse rows, printed per chunk.
- Each chunk is parsed into the last 256 KiB of its own 16 rows and widened in place (round 2):
  a row's int64 vectors end before input values not yet read. No separate input buffer.
- Output: `../fixed_width.hpp` per chunk of 2^16 values, so a newline ends every chunk
  (judge-specific; the checker compares tokens).
- Memory: 8 MiB + 16 KiB int64 + 4 MiB dwords: 6 huge pages and 4 small ones below them. a's
  dwords are stored strip after strip (8 KiB each, no padding); output chunks reuse a's space.
- Runs from `.preinit_array` and ends with `_exit` (as `convolution_mod`).

## Measurements

AMD EPYC 7B13 (`lc-amd`), GCC 15.2, judge flags, 2026-10-09. Whole process, judge runner,
max_random_00, median of 21 unless noted.

| Program | ms |
|---|---|
| floor: parse a and b only | 6.93 |
| floor: parse, print 2^20 values (fixed width) | 10.19 (11.4-11.8 when the VM was busy) |
| v1 | 18.55 |
| final (`judge.py bench`, 21 rounds, worst of the 3 max cases) | 15.77; v1 19.07, ratio 0.826 |

Floor phases: parse 3.68, print 4.55 (mostly `write(2)` of 10 MB). Final phases (in-process):
rows a 2.72, columns a 0.60, rows b 2.31, columns b 1.04, inverse rows + print 5.34.
Column sub-steps: a transform 0.25, a mod P 0.35 (includes faulting its 4 MiB), b transform 0.27,
products 0.56, inverse 0.25.

Zen 3 (`lc-amd`), dependency-free instructions per cycle: `vpaddq` 4, `vpmuludq` 2, `vpsrlq` 2,
`vaddpd` 2, `vfmadd` 2, `vpunpcklqdq` 2, `vperm2i128` 1.0, `vpermq` 0.79; mixes: mul+add 3.0,
mul+add+add 4.0, mul+srl 4.0, mul+fadd 2.9. 256-bit stores 1 per cycle.
Page faults: 16 MiB in 2 MiB pages 0.73-0.85 ms, in 4 KiB pages 6.7 ms; whole process +0.044 ms
per MiB of huge pages.

## Log

2026-10-09, claude, round 1 (`lc-amd` unless noted; same-run medians):
- v1: int64 lanes, rows of 2^12, 4096-token parses, column pass in strips with a separate product
  loop, output array then `fixed_width::write`. 13/13 tests, 18.55 ms. Phases: parse+rows 7.20,
  columns 2.08, inverse rows 1.02, print 4.58. Parse alone in 4096-token calls: 5.23 vs 3.68.
- v2: parse in 2^16-token chunks; b's last column levels, the products and the first inverse
  levels in one kernel. 18.98 -> 17.62. Rows 7.20 -> 6.10; the fused kernel lost (columns
  2.08 -> 2.24: GCC spills it).
- v3: radix-16 in assembly (slot store at every level); products 4 at a time; fused kernel
  removed. 17.19 -> 16.48. A shift-based scale was wrong (two reductions divide by 2^64);
  caught by the stress test.
- v4: a transformed completely first and kept as dwords; b reuses the int64 array; output per
  chunk. 16.44 -> 16.31 (footprint 17 -> 12.6 MiB).
- v6: transpose loops were not unrolled (`v[4]` on the stack, copy loops) and GCC reassociated
  the radix-4 sums (10 ops for 8): `#pragma GCC unroll` and an empty asm after each level.
  Radix-16 assembly rescheduled ("late spill"). 16.40 -> 15.45.
  Kernels, TSC cycles per vector for 8 bits (two radix-16 passes, L1): GCC radix-16 4.06,
  GCC radix-8 3.00, assembly with a slot per level 3.00, all in place (88 ops) 2.54, late spill
  2.37, late in place (71 ops) 2.38.
- v7: reductions and products stage by stage over 4 vectors. 15.43 -> 15.29. Products in L1:
  6.45 core cycles per vector (7 multiplies; the mul pipes are the limit).
- No gain: rows of 2^11, 2^13, 2^14 (16.1, 16.0, 16.0 vs 15.6); `MADV_POPULATE_WRITE` (15.34 vs
  15.37); a's dwords without `vpermq` (15.39 vs 15.34); 8 products in flight (L1: 6.1 vs 5.6 TSC
  cycles per vector, spills).
- `perf` on `lc-intel` (v4, user cycles): parse 41%, radix passes 15.5%, inverse rows 9%,
  formatting 8.6%, forward rows 8%, products 7.7%, a mod P 4.1%.
- Checks: 13/13 official tests (`lc-amd`, `lc-intel`); `stress.py` 200 rounds against `brute.cpp`
  (N <= 12) plus three N = 20 inputs with known answers (all P - 1; a = P - 1; b a delta), which
  hit the 2^51 bound; ASan/UBSan build on all official tests and 60 random inputs.
- PR #45 merged (CI: 13/13 on 3 runners, slowest 12.7-15.6 ms). Submitted its `main.cpp`:
  [409198](https://judge.yosupo.jp/submission/409198), AC, 15 ms, 23.4 MiB (1/5). Previous
  record 25 ms.

2026-10-09, claude, round 2 (`lc-amd`, judge image and flags). "Probe": whole-process wall time
and in-process `CLOCK_MONOTONIC` phase stamps, max_random_00, programs interleaved, medians of
21-31 runs. Phases of round 1 (ms): start 1.09, parse 3.77, rows 1.26 (incl. 0.4 of x's page
faults), columns a 0.59, columns b 1.05, inverse rows 0.75, print 4.63, exit 0.94; total 14.13.
- `.preinit_array` start: start 1.23 -> 1.12, total ratio 0.990. Kept.
- Products in doubles with a scaled by 2^-N in doubles too: columns b 1.04 -> 0.92, columns a
  0.60 -> 0.70; total ratio 1.001. Scale by Montgomery with R = 2^N instead (5 integer ops, as
  cheap as the old reduction): columns a 0.59, columns b 0.92; ratio 0.990. Kept.
- In-place parse, a stored strip after strip, 6 huge pages instead of 7: rows a 2.74 -> 2.56,
  columns b 0.92 -> 0.85 (a read sequentially); ratio 0.976. Kept.
- Reduction keeps only the low dword (no `vpsubq` of the magic bits), `vinserti128` for two of
  the four 128-bit permutes in the transpose: inverse rows 0.76 -> 0.73; total ratio 1.0005. Kept.
- No gain: row pieces of 64 vectors (lanes and 4 vector bits per 2 KiB piece, then the rest of
  the row; 14.18 vs 14.13); chunks of 2^17, 2^18, 2^20 tokens (ratios 0.997, 1.010, 0.999);
  prefetching the next row in the inverse rows, T0 or T1 (inverse 0.73 -> 0.78).
- Probe totals: round 1 14.13 -> 13.45 ms. `judge.py bench`, 21 rounds, worst of the 3 max
  cases: round 1 15.68, round 2 15.06, ratio 0.960.
- Checks: 13/13 official tests (`lc-intel`); `stress.py` 200 rounds plus the three N = 20
  known-answer cases (judge image on `lc-amd`); ASan/UBSan on all 13 official tests and a pipe.
- Compute left (probe): rows ~0.48 per array, columns a 0.57 (0.18 of it page faults), columns
  b 0.85, inverse rows 0.73. Transforms run ~1.4 instructions per vector-level against a floor of
  ~1.25 for radix-16 in 16 registers; the rest is the I/O floor and process start/exit.
- PR #53 merged. CI ratios: EPYC 7763 0.973, EPYC 9V74 0.963 and 0.931. Submitted its `main.cpp`:
  [409210](https://judge.yosupo.jp/submission/409210), AC, 14 ms, 23.0 MiB (2/5).
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  14.97 → 14.68 ms (0.981). 13/13 official tests. ASan/UBSan on the 3 largest cases, file and pipe.

## Next

- Compute left over the floor: ~4 ms of ~15.3. Largest: column pass 1.6 ms (products 0.56),
  inverse rows ~0.8, row transposes (~2.6 core cycles per vector forward, ~6 with the reduction
  in the inverse; `vperm2i128`/`vpermq` are slow on Zen 3).
- The inverse rows' last pass (transpose, reduction, pack) costs ~0.25 ms more than the forward
  first pass; not explained by its op count (guess: latency of the reduction chain).
- The floor itself (parse 3.7 ms, `write(2)` ~4.5 ms) belongs to `lib/io`.

## Sources

- Walsh-Hadamard transform for xor convolution: https://en.wikipedia.org/wiki/Hadamard_transform
- Montgomery reduction: P. L. Montgomery, Modular multiplication without trial division, Math.
  Comp. 44 (1985).
- int64/double conversion with the 1.5 * 2^52 constant: https://stackoverflow.com/questions/41144668
- Exact product as h + l with FMA (error-free transformation): standard; e.g. Ogita, Rump, Oishi,
  Accurate sum and dot product, SIAM J. Sci. Comput. 26 (2005).
- `.preinit_array` start: taken from `../convolution_mod/solution.cpp`.
- Zen 3 costs: measured here; see AGENTS.md for uops.info.
