# bitwise_xor_convolution

N <= 20; a, b of 2^N values < 998244353; print c_k = sum over i xor j = k of a_i b_j mod P. 5 s.
Input ~20.7 MB (2^21 tokens), output ~10.4 MB.

Best judged: ours, [409198](https://judge.yosupo.jp/submission/409198), 15 ms (round 1).
Record when opened: 25 ms.

## Design

- c = H (H a . H b) / 2^N, H the Walsh-Hadamard transform: 3 transforms of 20 levels.
- Signed 64-bit lanes, no reduction inside a transform: |x| <= 2^20 * 1.6 P < 2^51.
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
  Products: two Montgomery reductions, the second by 2^(64-N) mod P.
- Order: a's rows while it is parsed (chunks of 2^16 tokens: smaller bulk reads leave ~1000
  tokens per call to the scalar path), a's columns, a mod P kept as dwords (4 MiB); b reuses the
  int64 array: rows, columns with the products, inverse columns; inverse rows, printed per chunk.
- Output: `../fixed_width.hpp` per chunk of 2^16 values, so a newline ends every chunk
  (judge-specific; the checker compares tokens).
- Memory: 8.4 MiB int64 + 4.2 MiB dwords + 256 KiB, one mapping in 2 MiB pages.

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

## Next

- Compute left over the floor: ~4 ms of ~15.3. Largest: column pass 1.6 ms (products 0.56),
  inverse rows ~0.8, row transposes (~2.6 core cycles per vector forward, ~6 with the reduction
  in the inverse; `vperm2i128`/`vpermq` are slow on Zen 3).
- Products: 7 multiplies per vector. Folding 2^(32-N) into a's dwords moves work, not removes it.
- The floor itself (parse 3.7 ms, `write(2)` ~4.5 ms) belongs to `lib/io`.

## Sources

- Walsh-Hadamard transform for xor convolution: https://en.wikipedia.org/wiki/Hadamard_transform
- Montgomery reduction: P. L. Montgomery, Modular multiplication without trial division, Math.
  Comp. 44 (1985).
- int64/double conversion with the 1.5 * 2^52 constant: https://stackoverflow.com/questions/41144668
- Zen 3 costs: measured here; see AGENTS.md for uops.info.
