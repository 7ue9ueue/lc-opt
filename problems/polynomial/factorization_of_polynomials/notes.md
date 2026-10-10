# factorization_of_polynomials

Factor a monic f of degree N <= 100 over F_p, p prime <= 998244353, into monic irreducibles with
multiplicities. One polynomial per test, 10 s. Tests (67): examples, constant, all_linear,
small_factors (random p, factors of degree <= 4, multiplicity <= 3), random_coeff, irreducible_1/2/3
(one to three shifted binomials (x + b)^d - a of degree up to 100, p in {101, 998244341} for d = 100),
same_degree (p in {101, 998244353}, k factors of one degree d <= 12), square_free (+ consecutive
degrees 1..13, sparse degrees 1, 3, 7, 15, 31), large_p2 (p = 2, degree 100), zero_derivative and
multiplicity_multiple_of_p (p <= 7). The checker tests irreducibility itself.

Best judged: 9 ms, clean 1 ms ([409665](https://judge.yosupo.jp/submission/409665),
[409666](https://judge.yosupo.jp/submission/409666), #323). Record when opened (issue #83): 161 ms.

## Design

- p = 2 (`gf2.hpp`): polynomials as 128-bit masks; Berlekamp with Q rows x^(2 j) mod g by shifts,
  the kernel by elimination on bit rows, and pieces refined by gcd(h, b) for every kernel vector b.
- Odd p (`field.hpp`, `poly.hpp`, `kernel.hpp`, `factor.hpp`): Montgomery residues (2^32), u64 lazy
  sums folded every `chunk` products (14 for p ~ 2^30), one reduction per output.
  - Square-free decomposition (Yun, p-th roots where f' = 0); a square-free f skips the loop.
  - Per square-free part g of degree m: s = x^((p-1)/2) mod g (squarings: product, then
    reduction by the table x^(m + j) mod g), x^p = x s^2. Linear factors gcd(g, x^p - x) by
    Cantor-Zassenhaus root finding, s giving the first split; p <= 256 by evaluation.
  - Rest by Berlekamp: rows q_j = x^(p j) mod g by mat-vecs with the matrix of the
    multiplication by x^p (columns x^s r mod g), 8 rows at a time transposed into A = (Q - I)^T;
    the kernel of A (dimension k = number of factors) by fraction-free blocked elimination
    (`kernel.hpp`: 16-column blocks, panel and pivot rows and trailing rows each one lazy linear
    combination, one batched inverse at the end).
  - k > 1: v and w random kernel elements (back substitution with random free entries). The
    factors are grouped by the value of v (v = c_i mod g_i): the c_i are the roots of the minimal
    polynomial mu of v, found from the coordinates of v^j (entries at the free columns, a k x (k + 1)
    kernel). Groups by divide and conquer over the roots: gcd(h, P(v)), P(v) from the powers of v
    near the top, by Horner on v mod h below. For large p all groups are single factors but for
    probability k^2 / 2p; a group where w is not constant is factored again (Berlekamp on it), and
    if the count of factors still falls short of k all groups left are.
  - Degree <= 8 moduli: polynomials in one register (`multiply_small`), as root finding runs
    many chains of 30 dependent squarings at tiny degrees.
  - Memory: page faults cost ~1.6 us per 4 KiB page (`faultbench.cpp`, lc-bench), so the two
    m x m arrays reuse their storage: the reduction table, then the multiplication matrices (by
    x^p, by v); A, then the powers of v.

## Floor

`tools/speed.py bench` on `lc-bench` (EPYC 7B13): the C floor (tools/floor.c) 0.70 ms on every
test; our binary on the trivial tests (constant_00, example_00, small p = 2 cases) 1.02-1.04 ms:
loading libstdc++ (the judge's link has no --as-needed). So ~1.02 ms is the practical floor.

## Measurements (round 1, final)

`speed.py bench`, `lc-bench`, 11 rounds: score 1.406 ms (same_degree_03 1.406, irreducible_2_02
1.350, irreducible_2_01 1.341, irreducible_2_04 1.325, irreducible_3_01 1.318, ...; all others
<= 1.31). Minor faults: 106 (example_00), 135 (irreducible_1_01), 138 (same_degree_03).

In process (`trace.cpp`, `lc-bench`, median us):

| step | irreducible_1_01 (m 100, p 998244341, k 1) | same_degree_03 (m 88, p 998244353, k 44) |
|---|---|---|
| x^((p-1)/2) mod g | 43 | 31 |
| Krylov (99 mat-vecs) + transposes | 86 | 49 |
| kernel of (Q - I)^T | 57 | 36 |
| split (mu: powers 22, roots of mu 77, groups 35) | - | 153 |
| total | 209 | 294 |

Machine facts (`mulbench.cpp`, `faultbench.cpp`, lc-amd/lc-bench): vpmuludq ymm 2 per cycle on
Zen 3 (also vpmaddwd, vpmulld, vfmadd231pd); vpaddq 4 per cycle. The mat-vec kernel (`combine`)
reaches ~4.5 products per cycle at any size (L1 or L2): its bound is the FP ports (8 mul, 4 shift,
8 add per 32 products), not memory. A 4 KiB page fault 1.6 us, MAP_POPULATE 1.4-2.7 us per page.

## Log

- 2026-10-10, claude (round 1): first solution, Berlekamp as above.
  - v1: all 67 official tests and 300 stress rounds; `speed.py` score 1.70 ms (same_degree_03).
    Two TLEs on the way: C-Z on B with v lacking a random constant never splits k = 2 (the ratio
    of the two values is fixed).
  - combine_block: GCC kept the accumulator arrays in memory (load-add-store per product);
    `#pragma GCC unroll` and folds outside the column loop gave a clean register loop. Krylov
    126 -> 89 us (m 100).
  - Elimination: lazy u64 rows (80 KiB, L2-bound: 194K TSC in row updates) 90 us; blocked u32
    with Gauss-Jordan pivot rows by lincomb 74 us; fraction-free blocked with lazy combines (no
    inverses) 57-62 us at m 100.
  - Splitting many factors: C-Z on the kernel algebra 480 us for k 44 against 306 for the
    minimal polynomial route; Tonelli-Shanks (s = 23 for 998244353) replaced by Cipolla
    (0.42 us); scalar/vector paths for degree <= 8 (root finding of mu 108 -> 77 us); groups pass
    v mod h down (extraction 63 -> 35 us). Small p (101) now also takes the minimal polynomial
    route, collisions refactored: square_free_consecutive_degrees_02 256 -> 207 us.
  - Memory: 163 -> 135 faults (irreducible_1_01) by reusing the m x m storage and no full kernel
    basis (k vectors) nor vector of Poly when every group is one factor.
  - Checks: 67/67 official tests (`judge.py test`, lc-amd); `stress.py` 2000 rounds against
    `brute.cpp` (DDF + EDF on f itself, no square-free step); ASan/UBSan build on all official
    tests, outputs compared as sorted factor lines.
  - Submissions (#323's main.cpp): 409665 AC 9 ms, 409666 AC 9 ms. Both clean 1 ms
    (`spikes.py`): launch spikes of 8-9 ms on 4 and 1 cases whose peers ran in 0-1 ms. Every
    real case is judged 0 or 1 ms. P(clean run) = 0.03 with 67 cases, so 3 submissions are kept.
  - Next: same_degree_03's split (root finding of a degree-44 mu: ~40 small nodes, each a chain
    of 30 dependent squarings; batch them across nodes), the dense kernels at ~4.5 of ~6.4
    products per cycle (FP-port bound), x^((p-1)/2) (symmetric squaring), page faults (~30
    pages over the trivial tests).
- 2026-10-10, audit (claude): submissions not logged before; who submitted them is not recorded.
  22:05-22:13 UTC, after the issue closed; the folder last changed in #326, so #323's `main.cpp`
  (guess). [409699](https://judge.yosupo.jp/submission/409699) 9 ms,
  [409700](https://judge.yosupo.jp/submission/409700) 10, [409701](https://judge.yosupo.jp/submission/409701) 11,
  [409702](https://judge.yosupo.jp/submission/409702) 9, [409703](https://judge.yosupo.jp/submission/409703) 10,
  [409704](https://judge.yosupo.jp/submission/409704) 10; all AC, all clean 1 ms (`tools/spikes.py`).
  Best judged stays 9 ms.

## Sources

- E. R. Berlekamp, "Factoring polynomials over large finite fields", Math. Comp. 24 (1970): the
  kernel of Q - I, and splitting with roots of the minimal polynomial of a kernel element.
- D. G. Cantor, H. Zassenhaus, "A new algorithm for factoring polynomials over finite fields",
  Math. Comp. 36 (1981): gcd(h, v^((p-1)/2) - 1).
- D. Y. Y. Yun, "On square-free decomposition algorithms" (1976), and the characteristic-p form
  (p-th roots) as in the Wikipedia article "Factorization of polynomials over finite fields".
- M. Cipolla (1903) square roots; P. L. Montgomery, "Modular multiplication without trial
  division", Math. Comp. 44 (1985).
- No code was read; nothing from lib/ but io and run (no lib/poly change).
