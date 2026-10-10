# polynomial_root_finding

All roots in F_p of f, deg f = N <= 4000, p = 998244353, distinct, any order. 10 s. Tests:
examples (4), deg0 (N = 0), small_random (N <= 10), and N = 4000: all_distinct (4000 distinct
roots), all_same ((x - a)^4000), max_random (m random linear factors over k values times a random
polynomial of degree 4000 - m; odd seeds repeat roots: 03 has 1158 roots and 920 multiple zeros
on H, nearly all repeated roots; 05 has 1186 and 1023). The checker compares root multisets.

Best judged: none yet. Record when opened (issue #85): 151 ms.

## Design

One round of the tangent Graeffe method (Grenet, van der Hoeven and Lecerf), with multiple zeros
resolved by climbing back through saved iterates instead of new rounds.

- Roots 0 come off first (f_0 = 0). g = f(x + t) for a random t with g(0) != 0 (one cyclic
  product of length 2^13; factorials in vector blocks).
- Graeffe steps on (A, B) = (g, g') over F_p[eps]/(eps^2): A(y) + eps B(y) =
  (A + eps B)(x) (A + eps B)(-x), y = x^2 (graeffe.hpp). In the leaf domain of lib/poly's
  transform: leaves 2q and 2q + 1 of the length-2L transform (weights +-r) give leaf q of the
  length-L transform of the result (CRT of the two even parts mod y^4 -+ r). Per step: the leaf
  pass, two inverses and two forward_upper of length L = 4096. B carries 2^-k, so a simple zero z
  of A_k gives its root r = z A'(z) / B(z) directly.
- After k = 23 - l steps (l = bit_width(N) - 3, 9 at N = 4000) the roots in F_p sit in H, the
  subgroup of order 119 2^l (60928 points); other roots never land there. H = 119 cosets of the
  2^l-th roots of unity, 8 cosets in the lanes of a vector: fold each polynomial mod
  (x^n - v^n) (64-bit sums, one Montgomery reduction per output), twist by v^j, then
  Transform's forward in the lanes layout (outputs at w^bitrev(q)). A, xA' and B on H.
- A zero z of A on H with xA'(z) != 0 is simple: its root. Otherwise z is multiple (roots that
  collide under r -> r^N, or a repeated root) and climbs: its preimages under x -> x^(2^e) are
  evaluated with the iterate saved at level k - e (A, xA', B), then at level 0 with g itself,
  where every zero among the candidates is a root (multiplicity no longer matters). k = 14:
  levels 14 -> 7 -> 0, 128 candidates per multiple zero each time. A tracked point whose
  candidates hold one zero must send all its roots there, so that zero is multiple without
  evaluating xA' and B; only lane groups with two zeros in a lane evaluate them.
- Points are exponents of the generator 3 (mod p - 1), so the 2^e-th roots of a tracked point are
  exact divisions by 2^e; coset bases are recomputed by an 8-lane exponentiation.

Costs at N = 4000 (`lc-bench`, in process, us): shift 35-45, Graeffe 335-350 (14 steps: the leaf
pass ~55% of a step, transforms the rest), H 300-375, climb 220 (all_distinct, 137 multiple
zeros, all collisions) to 1100 (max_random_05, 1023 multiple zeros, nearly all repeated roots:
two folds of all N coefficients per root), init 100-160 (one huge page, tables), read ~60.

## Floor

`tools/speed.py bench` on `lc-bench` (9 rounds, main 2cee7dab): floor (a C program mapping the
input and writing the output) 0.69-0.75 ms on every test. deg0, where solve() does nothing but
I/O, takes 1.03-1.06 ms: every C++ program loads libstdc++ (the judge's link has no
--as-needed; an empty C++ program also lists libstdc++.so.6 as NEEDED), so ~1.0 ms is the
practical floor here.

## Log

- 2026-10-10, claude (round 1): first solution.
  - v1 (dc54577): rounds of shift, Graeffe and evaluation on H; found roots divided out by a
    product tree (PointTree) and a power series division; on repeated roots (more multiple zeros
    than collisions predict, or any in round 2) f / gcd(f, f') from the half-gcd jump's last
    row (lib/poly/gcd.hpp, R[1][1]). 36/36 tests (`judge.py test`, `lc-amd`), slowest 4.0 ms
    (max_random_07). Profile: a round at N ~ 4000 cost ~0.75 ms (Graeffe 360 us, evaluation
    300 us), the radical 0.4-0.66 ms, and max_random tests need up to 3 rounds (their junk
    keeps the degree up after deflation).
  - v2 (f522800): one round; multiple zeros climb through saved iterates (Design). No
    deflation, no gcd. max_random 2.0-2.1 ms, but max_random_03/05 4.2-4.4 ms: ~1000 repeated
    roots, each climbed with three polynomials.
  - Coset kernel: per-polynomial sums held in registers (GCC kept them in memory: load-add-store
    per product), only A at intermediate climbs unless a lane has two zeros, one Montgomery
    reduction per output (chunks of 16 products folded first), two outputs per pass for one
    polynomial, the twist chain split. evaluate of 3 polynomials on 8 cosets of 512 points:
    91K -> 52K TSC cycles; 1 polynomial on 128 points (the climb): 12K -> 9.4K, near the
    multiplier bound (Zen 3 issues one ymm vpmuludq per cycle; 2 per 8-lane multiply-add).
  - Shift: vector Montgomery products and factorial tables from in-vector prefix products.
  - Tried, no gain: l = 8 at N = 4000 (n = 256 points per coset, one more Graeffe step):
    junk-heavy max_random 1.88 -> 1.80 ms, all_distinct 2.14 -> 2.29, max_random_05 2.89 ->
    3.05 (`speed.py`, `lc-bench`). Four outputs per pass: register spills, slower.
  - Checks: 36/36 official tests (`judge.py test`, `lc-amd`); `stress.py` 400 rounds against
    `brute.cpp` (N <= 300: distinct, repeated, one high-multiplicity root, roots with many
    collisions, junk, small alphabets, N = 0..3) and 15 rounds at N in [500, 2047]; ASan/UBSan
    build on all official tests (root sets compared).
  - `speed.py bench` (`lc-bench`, 9 rounds, main 2cee7dab): max_random_05 2.90 ms, 03 2.77,
    07 2.33, all_distinct 2.12-2.15, all_same 1.87, other max_random 1.84-1.99, small 1.04-1.16.
  - Next: the repeated-root climb (1.1 ms on max_random_05; the radical route costs about the
    same, ~0.6 ms jump + a second pipeline at N ~ 1200, unless repeats are detected before the
    Graeffe pass); the leaf pass (~40K TSC cycles per step against ~400 multiplies' worth);
    H's three transforms per group (~45% of H); init (one 2 MiB huge page, tables).

## Sources

- B. Grenet, J. van der Hoeven, G. Lecerf, "Randomized root finding over finite FFT-fields using
  tangent Graeffe transforms", ISSAC 2015: Graeffe transforms over FFT primes, the tangent
  (F_p[eps]) trick to recover roots, evaluation on the subgroup. Their collisions are handled by
  new shifts; the climb through saved iterates is ours.
- J. van der Hoeven, M. Monagan, "Implementing the tangent Graeffe root finding method", ICMS
  2020: the shape of an implementation (Graeffe steps by FFT, the subgroup evaluation).
- No code was read; lib/poly supplies the transforms (Transform, lanes layout), derivative,
  Montgomery helpers and transpose8.
