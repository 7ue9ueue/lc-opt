# polynomial_interpolation

N <= 2^17 distinct points x_i and values y_i mod 998244353; print the N coefficients of the f of
degree < N with f(x_i) = y_i. 10 s. Slowest tests: max_random_00/01 and x_zero_00 (N = 2^17,
x_zero has one point 0), random_00 (N = 127669: the same transform lengths).

Best judged: none yet.
Record when opened (issue #77): 55 ms.

## Design

- `poly::interpolate` (lib/poly/interpolation.hpp, design in lib/poly/notes.md, Interpolation):
  PointTree's product tree of Q = prod (1 - x_i z) (lib/poly/evaluation.hpp); M'(x_i) by its
  transposed descent from the window rev(M') / Q (one Karp-Markstein division, as
  multipoint_evaluation); weights y_i / M'(x_i); then R = sum_i w_i Q / (1 - x_i z) by the
  transpose of the descent, whose coefficients reversed are f. The lane tree runs the descent and
  the sums in one depth-first pass; its base (32 slots) inverts the values by Montgomery's batch
  inversion and builds the sums from the same schoolbook products.
- I/O: `io::read_bulk` for the 2N tokens; output in 10-byte fixed-width fields
  (problems/convolution/convolution_mod/fields.hpp, judge-specific). `RUN_EARLY` (lib/run).

## Floor

`lc-amd`, whole process, judge-like runner (scratch `timeit.py`: judge.py's build and runner
without the checker), 11 rounds interleaved: read and write only (`floor.cpp`: the same reader,
the N values y written back) 2.51 ms median over the slowest of max_random_00, max_random_01,
random_00, x_zero_00.

## Log

- 2026-10-10, claude (round 1): first solution and lib/poly/interpolation.hpp.
  - Checks: 9/9 official tests (`judge.py test`, `lc-amd`); ASan/UBSan on all 9, file and pipe
    input (`lc-intel`); `stress.py` 300 rounds against `brute.cpp` (N <= 1500; larger N up to 2^17
    by Horner at 8 points; points small with 0, near P - 1, roots of unity, sorted, random with
    0; values random, 0 / 1 / P - 1, zero, sparse); lib/poly tests at -O2 (native and
    x86-64-v3) and ASan/UBSan.
  - Versions, `lc-amd`, whole process, median of the slowest case (ms; floor 2.51): first
    (base sums by loops, one chain for the inversion) 19.97 (11 rounds). Then against it in the
    same run (15 rounds): unrolled and blocked sums kernels, inversion in 4 chains, buffers
    reused (lane windows' scratch, y with M', the window with R) 19.88 -> 19.63 (ratio 0.9875).
    The top leaves' descent converted to the lane root's state in the transform domain (lib/poly
    notes, Multipoint evaluation) 19.61 -> 19.54 (0.9970, 15 rounds).
  - CI of the first push of #257: multipoint_evaluation 0.9988, product_of_polynomial_sequence
    (byte-identical `.text`) 1.0019, verdict 1.0004 SLOWER; the conversion went into the same
    pull request.
  - Phases in process (`lc-amd`, max_random_00, fresh arena, ms), last version: tree 4.53
    (lane tree, top tree, tables, page faults), M' 0.08, division 2.22, descent and sums 9.49
    (top descent 1.79, lane pass 6.08 of which base 1.64, top sums 1.22, inverses 0.26; per lane
    level 0.22-0.31 descent and 0.20-0.30 sums). RSS 31.4 MB (floor 6.7 MB).
  - Base of 32 slots (rdtsc at 3.05 GHz TSC, `lc-amd`): 10797 -> 9394 cycles (descent alone
    4392). Sums levels: loops 641 / 640 / 702 / 946 / 1433 for D = 1 / 2 / 4 / 8 / 16; unrolled
    (D <= 8) 214 / 274 / 427 / 793 and blocks of 2 (D = 16) 1220. Blocks of 4 with zero padding
    for D = 2 .. 8 lost (640 / 823 / 1129: the padding's stores). Inversion: 1 chain ~1400, 4
    chains 1037 (the exponentiation alone 366).
  - Next: the build's lane root to the top leaves in the transform domain (the lane root's product
    transform gives the leaves' lower halves; needs leaves as transforms in ProductTree); a cheaper
    lanes-to-standard conversion would make the sums' one pay (0.081 ms now against 0.085 saved);
    the trees' scratch shared with the division's (~1-2 huge pages); the division's last forward
    of the window (half the input is q0's, whose transform the division has).
