# sqrt_of_formal_power_series_sparse

f with K <= 10 nonzero terms, N <= 10^6; print the first N coefficients of a square root of f
mod 998244353, or -1 if there is none. Any root is accepted. 10 s. Input is tiny; output up to
10 MB.

Best judged: none yet. Record when opened (issue #73): 49 ms.

## Tests

- small_dense_*: N = 10^6, i_k = 0 .. K - 1. A root exists for 00, 05, 06, 08 (K = 4, 9, 5, 6:
  recurrences of order w = 3, 8, 4, 5, all with slopes; dense output, 9.9 MB); 01, 02, 04, 07
  print -1 (f[0] not a square); 03, 09 are K = 0 (10^6 zeros). The slowest: 05 (w = 8).
- max_random_*: N = 10^6, K = 10, indices uniform. A root exists for 03 and 05 only (k = 68972,
  49248): long taps, output mostly zeros. The others print -1 (odd k or no square root).
- random_*: N up to 10^6, roots for 05, 07, 08 (long taps). small_N, min_K (K = 0), examples:
  tiny.

## Design

- f = a x^k (1 + h). No root if k is odd or a is not a square mod P (Euler's criterion), else
  g = x^(k/2) s with s = c (1 + h)^(1/2), c^2 = a (Tonelli and Shanks, as in
  sqrt_of_formal_power_series; the smaller root, so `brute.cpp` can match it). K = 0: g = 0.
- g^2 mod x^N depends on s mod x^(N - k) only, so s is solved to N - k coefficients and g's last
  k/2 coefficients are zero.
- (1 + h) s' = h' s / 2: n s[n] = sum_d ((3/2) d c_d - c_d n) s[n - d], c_d = f[k + d] / a: pow
  of a sparse series with M = 1/2. `poly::sparse::Holonomic` (`lib/poly/holonomic.hpp`), taps
  (d, (3/2) d c_d, -c_d); taps with d >= N - k are dropped.
- Output as in pow_of_formal_power_series_sparse: k/2 zeros as "0 ", s in chunks of 25600 in a
  ring (or one array if a tap reaches back further), groups of 16 zeros as "0 0 ... 0 ", other
  groups as fixed-width fields (`fields.hpp`), `write(2)` per chunk, then k/2 zeros; run from
  `.preinit_array`.

## Floor

`lc-amd`, small_dense_00, 05, 06, 08, whole process (judge's runner, 21 rounds, unchecked bench
in scratch), slowest case per round, medians: floor (read, fill 10^6 nonzero values, the same
output code) 6.59 ms; first version 8.96 (1.354 times the floor). Per case: floor 6.49 .. 6.51,
first version 8.49 (w = 3), 8.55 (w = 4), 8.63 (w = 5), 8.96 (w = 8).

## Log

- 2026-10-10, claude (round 1): first solution on `lib/poly/holonomic.hpp` as is.
  - Checks: 45/45 official tests (`judge.py test`, slowest 7.7 ms); `stress.py` 3000 rounds
    against `brute.cpp` (N up to 10^6; short, long, mixed and block-edge taps; odd and even
    leading zeros; f[k] a square or not; K = 0), and for N <= 64 both outputs squared against f;
    ASan/UBSan on all official cases, file and pipe input.
