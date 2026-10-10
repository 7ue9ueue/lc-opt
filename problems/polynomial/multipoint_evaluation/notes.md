# multipoint_evaluation

f(p_i) mod 998244353 for f of N <= 2^17 coefficients and M <= 2^17 points. 10 s. Slowest tests:
max_random_* and issue_1287_cpp_* (N = M = 2^17; issue_1287 has points that are roots of f) and
random_02 (N, M near 2^17).

Best judged: ours, 15 ms: [409516](https://judge.yosupo.jp/submission/409516) (`main.cpp` of #231),
clean score 15 ms (`tools/spikes.py`: no launch spike).
Record when opened (issue #75): 39 ms.

## Design

- `poly::evaluate` (lib/poly/evaluation.hpp, design in lib/poly/notes.md, Multipoint
  evaluation): the product tree of prod (1 - p_i x) in 8 lanes plus a top tree
  (lib/poly/product_tree.hpp) with every node's transform built in place; the root's window
  (x^(K-N) rev(f) / Q)[K - M8, K) by Karp and Markstein's division; then the transposed descent,
  halving each node's transform (2 transforms per level), schoolbook middle products below
  degree 32. Horner when N M <= 2^22.
- I/O: `io::read_bulk` for the N + M tokens; output in 10-byte fixed-width fields
  (problems/convolution/convolution_mod/fields.hpp, judge-specific). Runs from `.preinit_array`,
  ends with `_exit`.

## Floor

`lc-amd`, whole process, judge-like runner (scratch `timeit.py`: judge.py's build and runner
without the checker), 11 rounds interleaved: read and write only (`floor.cpp`: the same reader,
the M points written back) 2.67 ms median over the slowest of max_random_00, max_random_01,
issue_1287_cpp_00, random_02.

## Log

- 2026-10-10, claude (round 1): first solution and lib/poly/evaluation.hpp.
  - Checks: 11/11 official tests (`judge.py test`, `lc-amd`); ASan/UBSan on all 11, file and
    pipe input (`lc-intel`); `stress.py` 300 rounds against `brute.cpp` (N, M up to 9000: both
    paths; zero, repeated and P - 1 points; zero runs and P - 1 coefficients); lib/poly tests at
    -O2 and ASan/UBSan, native and x86-64-v3.
  - Versions, `lc-amd`, whole process, median of the slowest case (ms; floor 2.67): first (a
    forward of the window and an inverse per child at every node) 18.31 (9 rounds); halving
    descent, base level by level 16.76 (11 rounds). Then, each against the 16.76 version in the
    same run (`ab.sh`, 11-15 rounds): Karp-Markstein root division 0.9623; transforms built in
    place and the root's product transform reused 0.9320 (15.83); middle products in blocks
    0.9034 (17.11 -> 15.47); rebased on #232 (lib/poly transform changes) 0.9021 (16.63 ->
    15.00, 15 rounds); the division and the descent in one scratch (13 huge pages, was 14)
    0.9802 against the rebased version (15.03 -> 14.77, 21 rounds).
  - Phases in process (`lc-amd`, N = M = 2^17, fresh arena, ms), last version: lane tree 2.61,
    top tree 1.20, division ~2.3, descent 5.3 (top tree 1.38, of which leaf products 0.73; lanes
    2.80, of which base 0.68). 14 huge pages touched (~0.8 ms).
  - CI of #231: product_of_polynomial_sequence (re-bundled for product_tree.hpp; `.text` and
    `.rodata` byte-identical, `lc-intel`) timed 1.0019 (1.0065, 1.0014, 0.9977), then 1.0011
    (1.0023, 1.0055, 0.9956): noise failed the verdict twice; later commits went into the same
    pull request. Merged as #231; last CI: product_of_polynomial_sequence 0.9992 (0.9956, 1.0022,
    0.9999); multipoint_evaluation 11/11 on each machine, slowest 13.0-16.7 ms.
  - Submitted the merged `main.cpp` (#231): [409516](https://judge.yosupo.jp/submission/409516)
    AC 15 ms, 28.8 MiB; clean score 15 ms. Record when opened 39 ms.
  - Next: the trees' scratch shared (~1 huge page; needs ProductTree's split points kept apart);
    the conversions between the top tree's standard transforms and the lanes in the transform
    domain (3 butterfly levels inside leaves and 8 x 8 transposes instead of transforms of
    2^17-2^18 words, ~0.3-0.5 ms, estimate); the halving's combine pass fused into the next
    products (~0.08 ms). Considered: log.hpp's blocked division (B = 4) saves ~1.5 transforms
    of 2^17 against 1 more leaf product pass, ~0.1 ms.
