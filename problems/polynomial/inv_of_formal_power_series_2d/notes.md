# inv_of_formal_power_series_2d

N M <= 500000 coefficients a_ij of f(x, y) mod 998244353 (N rows of M), a_00 != 0; print
g = 1 / f mod (x^N, y^M) as N rows of M. 10 s. The checker compares tokens.

Tests (pinned commit, 28): 707 x 707; randomSquare 1045 x 478, 550 x 909, 440 x 1136, 545 x 917,
884 x 565; 10 x 50000 twice (degX10, and degX100, whose generator also makes 10 rows); 5000 x 100,
50000 x 10, 2 x 250000, 250000 x 2; 500000 x 1, 1 x 500000; maxDeg 389813 x 1, 463046 x 1,
53336 x 9; random 389813 x 1, 463046 x 1, 53336 x 6, 429249 x 1, 277012 x 1; five with N, M <= 10.

Best judged: ours, 28 ms with a +8 ms launch spike, clean 20 ms:
[409565](https://judge.yosupo.jp/submission/409565) (`main.cpp` of #276).
Record when opened (issue #88): 197 ms.

## Design

- `lib/poly/inverse_2d.hpp` (`Inverse2d`): Newton iteration in x over power series in y mod y^C,
  products by the Kronecker substitution y = z, x = z^S, S >= 2C - 1 (a product of two rows has
  y degree < 2C - 1, so nothing carries into the next row). Row 0 is the 1-D inverse of f(0, y).
  A step from k to k' rows is the 1-D inverse step (`inverse_step`) on this layout:
  e = f g_k mod (z^n - 1), rows k .. k' - 1 (each row's words past C cleared), then
  -(z^(kS) e) g_k mod (z^n - 1), rows k .. k' - 1, with n >= (k' - 1) S + C: terms that wrap land
  below row k. 5 transforms of length n per step, as in 1-D.
- Precisions N, ceil(N / 2), ..., 1, so each step's transform is the shortest power of two that
  holds it: about 2 N M words for the last step (2^20 for every large test but the 1-D ones,
  filled ~95%). Step 1 -> 2 is two products of single rows, of length >= 2C - 1 instead of 3C.
  S: the largest stride all steps allow at those lengths. Each step's products are computed a
  shift higher (z^a f g_k), so that the rows it wants lie in the upper half of the transform: its
  inverses compute only that half, and the second product's forward skips the zero lower half.
- Memory: the last step works in f's buffer (f is last read by its first product) and leaves its
  rows there (f is stored at that step's shift); g holds rows below that step's start; the
  scratch holds the transform of g_k and the earlier steps' work. 10 MB of arrays at 707 x 707
  (was 16).
- Few rows (3 <= R <= 32, when the cost model says so; on the tests R = 9 and 10, and 6 for
  53336 x 6): row by row in a transform domain of length n >= 2C - 1. g_i = -g_0 (sum over
  0 < t <= i of f_t g_(i-t)) mod y^C: the transforms F_t of f's rows and G_t of g's rows (each
  computed once), the sum as one inverse of a sum of leaf products (`detail::LeafProductSum`,
  a bottom for `inverse_with`), truncated, then a cyclic product with G_0. 5 (R - 1) transforms
  and (R - 1)(R + 2) / 2 leaf products of length n, against ~10 transforms of length 2RC.
- Orientation: the plan's cost model (transform word-levels and leaf products) picks Newton in x
  or in y (f transposed), and the method. A single row or column is the 1-D inverse.
- Cost against the 1-D inverse of N M coefficients: twice the transform length (the y padding of
  products truncated mod y^C), the same count. At 707 x 707: run 15.13 ms in process (`lc-bench`,
  warm), of which the last step (2^20) 8.04 (forward of g_k 1.30, product with f 3.36, product
  with e 3.28) and the step before (2^19) 3.76.
- I/O: `io::read_bulk` straight into f's buffer, rows then moved to their stride (last first);
  transposed orientations read into the scratch and transpose by 8 x 8 blocks (AVX2). Output in
  10-byte fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`, judge-specific),
  from the rows compacted into f's buffer (or transposed into the scratch). One `poly::Arena`;
  the program runs from `.preinit_array` (`lib/run/early.hpp`).

## Floor

`lc-bench`, whole process (`timeall.py`: judge flags, runner, 11 interleaved rounds): read and
write only (`floor.cpp`, read_bulk + fields) 4.23-4.35 ms on the large cases. First version
21.4-22.2 ms there. In process (first version, square_00): tables 0.15, read 0.85, scatter 0.45,
run 16.1-16.3 (cold), gather 0.33, write 0.33; total 18.4; the rest (~3 ms) is process start,
mapping and exit, as in the floor.

## Log

- 2026-10-10, claude (round 1): first solution, `lib/poly/inverse_2d.hpp` (owner lane of #95).
  - Checks: 28/28 official tests (`tools/judge.py test`, `lc-amd`); `stress.py` 300 rounds
    (N M <= 2000 against `brute.cpp`, shapes 1..10 x 1..10 first; larger shapes up to N M =
    500000 by f g = 1 at corners and random coefficients); ASan/UBSan on all 28 cases, file and
    pipe input; lib/poly tests (`test_inverse_2d`: every shape up to 12 x 12 against the
    recurrence, 14 shapes and their transposes to 3000 coefficients, 1 - x - y and 1, the test
    shapes and 1 x 2^19, 2^19 x 1) at -O2 -Wall -Wextra and ASan/UBSan. Mutations caught (each
    fails test_inverse_2d): no clearing of e's row tails, stride 2C - 2, no shift, the second
    product unshifted, the first step's upper half, transforms one row short, no negation.
    Checks repeated on the final version.
  - First version (dense input array, element-wise scatter and gather, stride 2C - 1 or the
    minimum over all steps): `lc-bench` 21.4-22.2 ms on the large cases.
  - Rows read in place and moved by `memmove`, 8 x 8 transposes, stride from the steps of at
    least 1/8 the longest length only (to make more steps meet k S >= n / 2): `judge.py bench`
    (`lc-bench`, 15 rounds, 6 slowest cases) 21.79 -> 21.30 ms (0.9808).
  - Last step in f's buffer, g only below its rows: 21.67 -> 21.19 ms (0.9811).
  - Upper halves by a shift instead of the stride: the stride rule above doubled the transforms
    of the steps below 1/8 (707 x 707: 6 -> 12 at 2^15, 23 -> 45 at 2^17; run 16.0 against 15.45
    ms). Stride from all steps again, each step shifted so its rows lie in the upper half (all
    steps qualify on the large tests): run 15.13 ms; whole process 21.19 -> 20.14 ms (0.9563).
    Upper against both halves alone (478 x 1045, run): 15.24 against 15.48 ms.
  - Considered (estimates in word-levels, 707 x 707): the tensor layout (y transforms once per
    row, x transforms on columns, all products pointwise in y-leaves) saves the y levels of every
    x transform, ~35% of the work with exact lengths, but needs power-of-two lengths in both
    variables: 2^21 words for 707 x 707 (Kronecker: 2^20), 268M word-levels against 197M;
    with a blocked last step 230M. Wins on shapes whose lengths round well (512 x 976: ~65M
    against 105M per last step). A truncated y transform (blocks y^1024 - 1 and y^512 - i,
    1536 points for 1413) would bring 707 x 707 to ~160M (guess). Blocked last step in the
    Kronecker layout (two halves at 2^19): 11 transforms and 7 leaf products of 2^19 against 5
    and 2 of 2^20, worse. A third-order last step from N/3: the step before stays at 2^19, no gain.
  - Merged as #276 (CI: correctness only, no baseline). Submitted its `main.cpp` twice:
    [409565](https://judge.yosupo.jp/submission/409565) AC 28 ms, 17.3 MiB (spike: randomSquare_03
    28, peers 20); [409566](https://judge.yosupo.jp/submission/409566) AC 28 ms (spikes:
    degX100_00 28, random_02 26, random_04 19). Clean score 20 ms both times (`tools/spikes.py`);
    16 cases within 9 ms of the max, so P(clean run) ~0.45.
  - Row by row for few rows (Design): `lc-bench`, `timeall.py` 11 rounds, whole process (ms),
    Newton -> row by row: degX10 19.97 -> 17.10, degX100 19.96 -> 17.04, degY10 20.32 -> 17.33,
    maxDeg_02 (53336 x 9) 20.27 -> 15.41, random_02 (53336 x 6) 17.15 -> 9.99; square_00 20.07,
    20.13 (unchanged). The cost model's estimates track the runs within ~5% (both methods, both
    orientations, 10 shapes; in process). Checks: 28/28 official tests, `stress.py` 300 rounds,
    ASan/UBSan of main.cpp on all cases and of the lib/poly tests (which run both methods on
    every shape with 3 <= rows <= 12 and on the large shapes with rows <= 16); 4 more mutations
    (row-by-row pairs shifted, no negation, one term short, the next group's windows from the
    current group) fail them.
- Next: the square-ish shapes and 5000 x 100 (~20 ms, all at a 2^20 last step) need another
  algorithm: the tensor layout (y transforms once per row, x transforms on columns; ~18% fewer
  word-levels at 707 x 707 with a truncated y transform of 1536 points and a blocked last step
  at X = 512, estimate). Row by row: F_t's windows [w a, a] stored once (each is used up to R - 1
  times; fill_windows is ~15% of a leaf product, guess).
