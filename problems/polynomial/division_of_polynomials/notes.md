# division_of_polynomials

f (N coefficients) and g (M coefficients, both with nonzero leading coefficient), N, M <= 500000,
mod 998244353: print q and r with f = q g + r, deg r < deg g. 10 s. With k = N - M + 1 quotient
coefficients and d = M - 1: the slowest tests have large k and d near k / 4 .. k
(n_max_02: k 446665, d 53335; max_random_02 / v_random_02: k 362953, d 92221; n_max_04: k 222989,
d 277011).

Best judged: ours, 12 ms: [409561](https://judge.yosupo.jp/submission/409561) (`main.cpp` of #271),
clean score 12 ms (`tools/spikes.py`).
Record when opened (issue #81): 30 ms.

## Design

- `poly::divide` (lib/poly/division.hpp, design in lib/poly/notes.md, Division): rev(q) = rev(f) /
  rev(g) mod x^k in blocks of s coefficients with h = 1 / rev(g) mod x^s; each block's residual from
  the previous block's last d coefficients by one cyclic product of length >= 2d (d < s), or from
  stored window transforms (log.hpp's blocks); the remainder from q g mod (x^L - 1), L >= d, or
  from the next block's residual. A cost model picks s and the remainder's method.
- I/O: `io::read_bulk`; output in 10-byte fixed-width fields
  (problems/convolution/convolution_mod/fields.hpp, judge-specific). `RUN_EARLY` (lib/run).
- q has exactly k coefficients (its leading one is f's over g's); r is trimmed.

## Floor

`tools/speed.py bench` on `lc-bench` (floor: map the input, write an output of the expected
size), 7 rounds, commit 63b9d82: n_max_04 3.89 ms, max_random_02 3.46, n_max_02 3.42; the
largest floor is n_max_01's 4.31 (its time 8.86).

## Log

- 2026-10-10, claude (round 1): first solution and lib/poly/division.hpp.
  - Checks: 35/35 official tests (`judge.py test`, `lc-amd`); ASan/UBSan build on all 35, file
    and pipe input (`lc-amd`); `stress.py` 300 rounds (`lc-intel`) and 200 (`lc-amd`) against
    `brute.cpp` (N, M <= 3000; up to 500000 by f = q g + r at random points); lib/poly tests at
    -O2 and ASan/UBSan (every block size and remainder method on 30+ size pairs).
  - Phases in process, cold (`lc-amd`, ms): n_max_04: read 1.31, setup 0.26, quotient 4.61,
    remainder 2.35 (split), write 0.42 (to /dev/null). n_max_02: read 1.00, quotient 6.18,
    remainder 0.50 (wrap). max_random_02: read 1.05, quotient 5.79, remainder 0.75.
  - Block size, in process, warm (`lc-amd`, ms; model in parentheses): n_max_02 (tails) 2^13
    10.17 (9.30), 2^14 7.89 (7.42), 2^15 6.26 (6.29, windows), 2^16 6.21 (6.13), 2^17 6.74
    (6.56), 2^18 8.28, 2^19 12.49; max_random_02 2^15 6.34, 2^16 6.35 (windows), 2^17 6.85
    (tails); n_max_04 2^15 6.32, 2^16 6.57, 2^17 6.96, 2^18 8.01; max_random_03 2^15 5.04,
    2^16 4.77 (tails), 2^17 5.85. The model picks the best or one within 1%.
  - Remainder (n_max_04): q g mod (x^(2^19) - 1) 6.74 ms in all, split (2^18 and a middle
    product for the top 14867 coefficients) 6.34. n_max_00 4.65 -> 4.33, n_max_01 3.33 -> 2.99.
    q0_equals_zero_01 (k = 2): directly 0.45, split 2.44.
  - One scratch span for the quotient and the remainder (fewer huge pages touched):
    `judge.py bench` on `lc-bench`, 15 rounds, slowest 3: 11.74 -> 11.44 ms (0.9746). The
    remainder from the next block's residual (kTail) and q written per block: 11.43 -> 11.42
    (0.9988, noise).
  - `speed.py bench` (`lc-bench`, 7 rounds): score 11.83 (n_max_04), max_random_02 11.52,
    n_max_02 11.30; compute (total - floor) 7.9-8.1 on all three.
  - Merged as #271 (CI: all tests pass; new problem, no timing comparison).
  - Submitted the merged `main.cpp`: [409561](https://judge.yosupo.jp/submission/409561) AC 12 ms,
    15.7 MiB; n_max_04 12, n_max_02 11, max_random_02 11, v_random_02 11, n_max_00 9;
    `spikes.py`: clean 12 ms. Record when opened 30 ms.
  - Next: n_max_04's windows by log.hpp's 2 x 2 Toeplitz products (~5 of 21 leaf products,
    ~0.3 ms); its remainder's top part from the quotient's stored transforms (the residual of the
    block after the last, ~0.3 ms); non-power-of-two block counts waste up to 46% of the first
    block (max_random_02: 6 blocks of 2^16 for 362953).
