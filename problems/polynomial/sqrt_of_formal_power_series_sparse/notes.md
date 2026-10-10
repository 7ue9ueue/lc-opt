# sqrt_of_formal_power_series_sparse

f with K <= 10 nonzero terms, N <= 10^6; print the first N coefficients of a square root of f
mod 998244353, or -1 if there is none. Any root is accepted. 10 s. Input is tiny; output up to
10 MB.

Best judged: ours, 9 ms: [409443](https://judge.yosupo.jp/submission/409443),
[409444](https://judge.yosupo.jp/submission/409444), [409446](https://judge.yosupo.jp/submission/409446)
(`main.cpp` of #206), each with launch spikes; clean score 7 ms (small_dense_00, 05 at 6-7 ms in
all three). Record when opened (issue #73): 49 ms.

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
  - Solve in process (`lc-amd`, 10^6 coefficients in chunks of 25600, a fresh `Holonomic` per
    run, medians of 11, repeated 3 times within 0.02 ms; small_dense 00, 06, 08, 05 = w 3, 4, 5,
    8): 2.09, 2.18, 2.27, 2.56.
  - Probes (same runs, variants of `holonomic.hpp`): without the block-to-block dependency 2.08,
    2.15, 2.23, 2.58 (no change: throughput-bound at every w); without the batch inversion of odd
    reciprocals 1.98, 2.03, 2.10, 2.37 (it costs 0.1-0.2); without the triangle G = F H and its
    two reductions 1.24, 1.32, 1.41, 1.69 (it costs 0.85, the largest part). Per block at w = 8
    the loop has 361 instructions (140 `vpmuludq`, 116 `vpaddq`); at w = 3, 280 (100, 76).
    Variants, bench and scripts: `~/explore/sqrt_of_formal_power_series_sparse` on `lc-amd`.
  - Not tried, estimated from op counts: the triangle in a layout of consecutive qwords
    (40 `vpmuludq` instead of 46) or as products of broadcast F[u] with windows of H (40, and
    12 shuffles instead of 24) both need y interleaved and G repacked, a net 4 to 12 fewer ops
    of about 400 per block; the batch inversion in qword lanes with lazy reduction, about 10
    fewer. Below 1% of the whole process each.
  - Merged as #206 (CI: new problem, all checks passed). Submitted:
    [409443](https://judge.yosupo.jp/submission/409443) AC 9 ms, 14.0 MiB: max_random_01 and
    small_N_00 9 ms, small_dense_04 8 (all three print -1) against 0-1 for their peers;
    small_dense_00, 05 7 ms. Resubmitted [409444](https://judge.yosupo.jp/submission/409444) AC
    9 ms: max_random_04 and small_N_00 9 ms (small_N_00 again; it takes the -1 path of the 0 ms
    cases), small_dense_08 8. Third: [409446](https://judge.yosupo.jp/submission/409446) AC 9 ms:
    max_random_08, 09 and small_N_06 9 ms; small_N_00 0 ms, small_dense_00, 05 7.
    `tools/spikes.py`: clean 7 ms for 409446; 9 for the first two, since small_N_00's reference
    is the median of a 9 and a 0. Not resubmitted again (3 of 5 used).
- Next: the solve is 2.6 ms of 9.0 at w = 8, throughput-bound, with the triangle the largest
  part (0.85 ms). A real gain needs fewer products per coefficient in G = F H, not a new layout.
