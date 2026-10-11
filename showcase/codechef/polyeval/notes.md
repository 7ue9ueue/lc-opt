# CodeChef POLYEVAL Evaluate the Polynomial

https://www.codechef.com/problems/POLYEVAL (JULY16, author sergey_adm). N, Q <= 2.5e5, TL 3 s,
modulus 786433 = 3 2^18 + 1. Source limit 50000 bytes (API field `source_sizelimit`).

## Statement (CodeChef API, `api/contests/PRACTICE/problems/POLYEVAL`)

- Input: N; the N + 1 coefficients a_0 .. a_N; Q; then Q lines, one point x_j each.
- Output: Q lines, the value at x_j mod 786433.
- 0 <= a_i, x_j < 786433; 0 <= N, Q <= 2.5e5 (subtask 1: <= 1000).
- The statement says "K different integers"; we do not rely on it (repeats tested).

## Routes

- Intended: every nonzero residue is a 3 2^18-th root of unity, so an NTT with one radix-3 step
  and 18 radix-2 steps evaluates f at all residues, O(p log p); then answer by table lookup.
- Ours: general multipoint evaluation. Subproduct tree of prod (x - x_i) down to blocks of at most
  64 points; remainder tree, each division by Newton inverse of the reversed divisor (one inverse
  per child, nothing cached); Horner on each block. O(n log^2 n).
- Products: 786433 is an NTT prime for lengths up to 2^18 (primitive root 10), so one cyclic
  convolution mod 786433 per product, on `multimod::Transform` with `Modulus(786433, 10)`. No CRT.
  Lengths up to 2^19 occur (quotient product when Q < ~120000).
- lib/easy is fixed to 998244353, so solution.cpp has its own small poly layer (multiply, cyclic,
  inverse, remainder) on lib/multimod.

## Evidence that the route normally fails

- https://discuss.codechef.com/t/can-someone-share-their-approach-to-polyeval/12741: a contestant
  implemented the O(n log^2 n) evaluation from the Waterloo CS487 notes and got TLE,
  "around 12 seconds for the worst case" (2016).
- Same thread: accepted approaches use the x^(2^k) orbit structure or the radix-3 NTT.

## Checks

- `lib/multimod` is tested only with primes near 2^30. explore `transform_test.cpp`: Transform mod
  786433 against a textbook NTT for lg 6..18, against chunked exact products for lg 19, 20, and
  at roots of unity for every lg; 6 shapes per lg (full, sparse, ragged tails, all p - 1). OK.
- `showcase/check.py`: sample OK, 1000 stress cases against brute.cpp OK (degree 0, Q = 0, repeated
  points, 0 and p - 1, sizes to 3000 to reach the transform path).
- ASan + UBSan (gcc:13, -O1, -Wall -Wextra clean): 80 small cases and max case 2 OK.
- baseline.cpp: 60 small cases against brute.cpp, all max cases equal to main.cpp.

## Results

2026-10-10, claude. lc-amd (EPYC 7B13, load 8-12 from other agents), gcc:13 `-std=c++20 -O2`,
explore `bench.py` (interleaved, each run alone under flock; the lock wait is not timed).

| Input (N, Q) | solution.cpp | baseline.cpp |
|---|---:|---:|
| 250000, 250000 | 312 ms median (298-321) | 2177 ms median (2102-2349) |
| 250000, 131073 | 202 ms (193-279) | 1489 ms (1382-1609) |
| 250000, 100000 | 159 ms (156-214) | 1177 ms (1165-1219) |

- 7 runs each. Ratio 7.0x on the worst case. Margin under the TL: 9.6x ours, 1.38x the baseline.
  An earlier 5-run session: 306 ms and 2205 ms.
- baseline.cpp: same algorithm and thresholds, textbook NTT mod 786433 (iterative radix-2, bit
  reversal, `% P`). Products longer than 2^18 split the longer factor (no 2^19-th root mod p).
  It fits 3 s on lc-amd, without the 2x margin. The 2016 contestant's version took 12 s (their
  code and machine unknown).
- Exploration files: `lc-opt-explore/showcase-polyeval/` (transform test, bench.py, run_all.sh).

## Judge caveats

- Source limit 50000 bytes: main.cpp is 114.6 KB (lib/multimod's generated kernels are 84 KB).
  It cannot be submitted to CodeChef as is. The same holds for every lib/easy bundle (about 410 KB).
- CodeChef compiler: assumed GCC 13, -O2, no -march; the pragma enables AVX2.
- `showcase/check.py` times `flock` together with the program, so its times include lock waits on a
  busy VM (first run: baseline "3.07-7.31 s" on a 190 ms case).
