# product_of_polynomial_sequence

N <= 500000 polynomials mod 998244353, degrees summing to D <= 500000 (leading coefficients
nonzero); print the D + 1 coefficients of their product. 10 s. Slowest tests: all_degree_one_*
(500000 linear factors, 10.9 MB of input) and max_random_* with large N (Poisson degrees, about
a third constants). unbalanced_* (one factor of degree ~450000) and max_and_zero (one factor,
the rest constants) are shallow.

Best judged: ours, 29 ms: [409448](https://judge.yosupo.jp/submission/409448) (`main.cpp` of #207),
clean score 29 ms (`tools/spikes.py`: no launch spike).
Record when opened (issue #74): 82 ms.

## Design

- Input: tokens bulk-parsed in rounds, each up to a lower bound on the token count (2 per
  polynomial not yet located); the walk that locates the polynomials also multiplies the
  constants (8 lanes of Montgomery products, 64 at a time) and counts degrees, branch free.
  Counting sort of the others by degree, largest first (degrees >= 256 by `std::sort`).
- Big polynomials (degree above 1/64 of the rest's) go to the top tree; so do all when fewer
  than 16 remain. The rest go to 8 lanes, polynomial 8k + l of the sorted order to lane l: one
  `poly::ProductTree<LaneLayout>` (lib/poly/product_tree.hpp) computes the 8 lane products
  side by side, with pointwise products. Slots of 8 polynomials of one degree load by gathers.
- The top tree (`StandardLayout`, leaf products) multiplies the lane products and the big ones.
  Both trees: transform doubling (one product pass, one inverse, one forward per node), the
  wrap trick (length = degree), Montgomery form throughout.
- Output in 10-byte fixed-width fields (problems/convolution/convolution_mod/fields.hpp,
  judge-specific). Two huge-page arenas, no heap arrays of size: on `lc-amd` a 4 KiB page fault
  costs 1.7 µs (0.43 ms per MB), a 2 MiB huge page 0.1 ms (0.05 ms per MB). The program runs
  from `.preinit_array` and ends with `_exit`.

## Floor

`lc-amd`, whole process, judge-like runner (scratch `timeit.py`: judge.py's build and runner
without the checker), 9 rounds interleaved: read and write only (`floor.cpp`: the same reader,
D + 1 values written) 7.86 ms median over the slowest of all_degree_one_00, max_random_01,
_03, random_01, unbalanced_00.

## Log

- 2026-10-10, claude (round 1): first solution, with `lib/poly/product_tree.hpp` (issue #95).
  - Checks: 27/27 official tests (`judge.py test`, `lc-amd` and `lc-intel`); `stress.py` 400
    rounds (`lc-intel`; D <= 3000 against `brute.cpp`, larger D by the product at 3 random
    points; linear factors, 2^k of them, random spreads, one large and many small, constants
    and one polynomial, equal degrees, N = 0); ASan/UBSan on all 27 official cases, file and
    pipe input; lib/poly tests at -O2 (native and x86-64-v3) and ASan/UBSan.
  - Versions, `lc-amd`, interleaved, 9 rounds, median of the slowest case (ms):
    first (lanes from single slots, separate heap vectors) 35.34; schoolbook base to degree 32,
    arena memory, branch-free walk, gathered slot loads 30.03 (ratio 0.854). Floor 7.86.
  - Phases of the current version, in process (`CLOCK_MONOTONIC` stamps, `lc-amd`, ms),
    all_degree_one_00: read 3.6, tables 0.26, lane tree 14.4, lane columns 0.3, top tree 5.7,
    scale 0.06, format 0.33; max_random_01: read 4.4, counting sort 0.9, lanes 14.5, top 5.6.
  - Lane tree by level (rdtsc, all_degree_one_00, per-lane length L, 65536 vectors per level):
    base (schoolbook to degree 32 and a forward of 64) 3.3; per level inverse (with products)
    0.35-0.66 and forward_upper 0.30-0.61, rising with L; root (L = 2^16) 0.8. Top tree:
    forwards of the 8 lane products at 2^17 words 0.9; levels 2^17 and 2^18 ~1.5 each; root
    (2^19) 1.0.
  - Microbenchmarks (`lc-amd`, rdtsc cycles): lane schoolbook 1x1 17, 2x2 31, 4x4 68, 8x8 196,
    16x16 734; forward / inverse of 64 vectors 816 / 813; a base node (32 linear slots, loads,
    products, forward) 5262, of which slot loads 1340 (scalar; gathers now). Per vector at
    2^17 words: lanes pointwise + inverse 30, forward_upper 28; standard leaf products +
    inverse 51: a standard level costs ~20 cycles per vector more than a lane level.
  - Tried, not kept: products fused into the inverse's bottom (each tile's products, then the
    asm bottom kernel from them): 1-1.5 cycles per vector slower than separate passes at every
    size (2^9 .. 2^20 words), in both layouts; base nodes built recursively by the tree's split
    (same time as consecutive pairs); base 16 instead of 32 (same); the counting sort's cursors
    in two interleaved chains (0.9 -> 1.3 ms); degrees kept as bytes for the sort (neutral).
  - Merged as #203 (CI: correctness only, no baseline).
  - The walk's `d ? 1 : tokens[at + 1]` (constants into the product) compiled to a branch:
    mispredicted for max_random's ~30% constants. As a mask: read phase of max_random_01
    4.2-4.5 -> 3.45-3.57 ms; whole process 30.25 -> 29.41 ms (ratio 0.972, `lc-amd`, 11
    rounds interleaved; all_degree_one unchanged at 28.7).
  - Counting sort placement alone (max_random_01, 305741 polynomials): 0.81 ms, 9.2 cycles
    each, plus a 0.1 ms huge-page fault for `order`. Lists for degrees 1-4 appended in the walk
    (a cursor array indexed by a degree table, the counting sort only for degrees 5-255):
    placement 0.15 ms but the walk +0.5 ms; whole process 0.993 (11 rounds), not kept. With
    five cursors and selects in registers instead: walk +1.7 ms (spills).
  - Merged as #207. CI: 0.9737 (EPYC 7763: 0.9717, 0.9746, 0.9749).
  - Submitted the merged `main.cpp` (#207): [409448](https://judge.yosupo.jp/submission/409448)
    AC 29 ms, 25.4 MiB; clean score 29 ms (13 cases within 9 ms of the max). Record 82 ms.
- Next: the transforms are ~2 per tree level (19 levels for linear factors) and near the
  kernels' speed; remaining overheads: the base (~1000 cycles per node of bookkeeping), the top
  tree's 3 standard levels (leaf products; a 2- or 4-coefficient leaf layout would cost less)
  and its 8 forwards from the lane root's coefficients (a lane forward_upper and an in-vector
  3-level inverse would give the standard transform directly, ~0.5 ms, guess).
