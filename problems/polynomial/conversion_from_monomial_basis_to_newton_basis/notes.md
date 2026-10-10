# conversion_from_monomial_basis_to_newton_basis

The b_i with f = sum_i b_i prod_(j < i) (x - p_j) mod 998244353 for f of N <= 2^17 coefficients
and points p_0 .. p_(N-1) (repeats allowed). 10 s. Slowest tests: max_random_* (N = 2^17), then
random_01 (N = 101656) and random_00 (N = 94808).

Best judged: none yet.
Record when opened (issue #84): 142 ms.

## Design

- `poly::to_newton` (lib/poly/newton.hpp, design in lib/poly/notes.md, Newton basis): with
  F = rev(f) and Q_t = prod_(i < t) (1 - p_i x), b_k = [x^(N-1-k)] F / Q_(k+1). The root's window
  F / Q mod x^N by evaluation.hpp's division, then a descent of the product tree of consecutive
  points: a right child's window is the low part of its parent's, a left child's a middle product
  with the right sibling (one product per node). Lanes hold consecutive blocks of points; only
  right children's transforms are kept; the lane count is padded to a power of two above 0.65.
- I/O: `io::read_bulk` for the 2N tokens; output in 10-byte fixed-width fields
  (problems/convolution/convolution_mod/fields.hpp, judge-specific). Runs from `.preinit_array`
  (`lib/run/early.hpp`). N = 0: an empty line.

## Floor

`lc-bench`, whole process, judge-like runner without the checker (scratch `timeit.py`), 21
rounds: read and write only (`floor.cpp`: the same reader, a_0 .. a_(N-1) written back) 2.23 ms
median over the slowest of max_random_00..02, random_00, random_01. `tools/speed.py` floor (no
parsing): 1.55 ms.

## Log

- 2026-10-10, claude (round 1): first solution, lib/poly/newton.hpp.
  - Checks: 15/15 official tests (`judge.py test`, `lc-amd`); ASan/UBSan main.cpp on all 15, file
    and pipe input (`lc-amd`); `stress.py` 300 rounds against `brute.cpp` (N up to 1500, larger N
    up to 2^17 by the identity at 3 points; repeated, 0 and P - 1 points); lib/poly tests (all) at
    -O2 and ASan/UBSan, Newton tests also at `-march=x86-64-v3`.
  - Versions, `lc-bench`, whole process, 21 rounds, slowest of the 5 slowest cases (ms; floor
    2.23): first (PointTree with points in lane order, right children without products) 13.90;
    then, each ratio against the first in the same run: the top tree's rightmost path from
    coefficients (no inverse) and node 0's base products skipped 0.9883; only right children's
    transforms kept and the build scratch reused for the division and the descent (RSS 32.8 ->
    18.9 MiB) 0.9271; lane count padded to a power of two above 0.65 (random_00, random_01 no
    longer slower than max_random) 0.9123, 12.68 ms.
  - Phases in process (`lc-bench`, N = 2^17, fresh arena, ms): trees 4.45, division 2.06,
    descent 3.9 (top tree 1.3, lanes 2.6 of which bases 0.54), output order 0.07. Profile
    (`lc-intel`, perf): transforms ~55%, leaf products 6%, page zeroing 5% (11% before the
    memory change).
  - Next: a division that keeps T(q0) for the root's right child (~0.07 ms, estimate); the
    blocked division (B = 4, ~0.1 ms); the top tree's leaves from the lane root's transform
    (~0.1 ms); the division's scratch partly in the tree's stack region (fewer huge pages).
