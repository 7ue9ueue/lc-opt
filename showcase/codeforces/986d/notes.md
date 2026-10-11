# CF 986D Perfect Encoding

- Problem: https://codeforces.com/problemset/problem/986/D (statement read on the Luogu mirror, CF986D).
- Limits: n has at most 1.5·10^6 decimal digits, no leading zeros. 2 s, 256 MB. Rated 3100, tags fft, math.
- Task: choose m >= 1 and b_1..b_m >= 1 with b_1·…·b_m >= n, minimizing the sum of the b_i.

## Intended solution

- The optimum uses 3s plus at most one 2 or one 4. With k least such that 3^k >= n, the answer is
  3k - 2 if 4·3^(k-2) >= n, else 3k - 1 if 2·3^(k-1) >= n, else 3k. n <= 4 gives n.
- The work is big-integer: 3^(k-2) in decimal, by FFT squaring, digits grouped into blocks.
- Evidence that ungrouped digits are expected to fail:
  - Per the coordinator's brief, the round's author (Um_nik, editorial blog entry 59720) wrote that digits
    must be grouped into blocks to pass with FFT. Not verified here: codeforces.com returns Cloudflare 403
    to fetches from this machine.
  - A Codeforces blog on a 63-bit NTT modulus ("Notes on FFT / NTT, and the 'ultimate' NTT with
    modulus > 9 * 10^18") uses 986D as its example: 6 digits per word "instead of the recommended 3",
    and calls the time limit tight (from a search-result summary; the page itself was not reachable).

## Our route

- `solution.cpp`: one decimal digit per coefficient, no packing. k from log3 n in doubles (leading 18
  digits plus the length; error < 1e-9, margin 1e-6). 3^(k-2) by binary powering: each step is
  `easy::multiply(x, x)` (a square: one forward transform) and a carry pass. Single prime 998244353:
  coefficients of the square are < 81·L < P. Then one check 9q >= n (bump k if not), and 4q, 6q against n.
- Squares: about 22, the last of 0.75·10^6 digits into a 2^21 transform. O(L log L).
- `baseline.cpp`: the same algorithm on a textbook NTT (iterative radix-2, bit reversal, `% P`).
- `brute.cpp`: exact search f(x) = min(x, min_b b + f(ceil(x/b))) for n <= 10^5 (checks the 3s-and-2-or-4
  shape); for larger n, the shape with exponents found by schoolbook ×3, O(L^2).
- `gen.py` max cases: 0 all 9s, 1 10^(L-1), 2 the largest 3^k with 1.5·10^6 digits (k = 3143854, exact,
  via Python's decimal), 3 random digits; all with L = 1.5·10^6. Small: 1..60, n <= 10^5, random up to
  3000 digits, c·3^k + {-1, 0, 1} for c in {1, 2, 4}, k <= 6000.

## Log

2026-10-10, showcase agent, lc-amd (EPYC 7B13), gcc:14 `-std=c++23 -O2` (Codeforces flags, no -march;
the pragma in lib/easy/poly.hpp gives AVX2). Times are wall time per process, interleaved, under one
`flock /tmp/bench.lock` (`bench.py` in the exploration folder). The check.py of that morning timed the
wait for the lock too (1.1-2.4 s medians while other agents benched); fixed since in 2219198.

| Version | max0 | max1 | max2 | max3 |
|---|---|---|---|---|
| first try: k from digit count, up to 4 fix-up ×3 steps, 64-bit carry | 114 | 100 | 132 | 134 |
| baseline, same first try | 501 | 481 | 538 | 540 |
| final: k from log3 n, 1 check, 32-bit carry | 76 | 76 | 75 | 70 |
| baseline, final | 459 | 460 | 462 | 451 |
| final, rebundled on lib/easy 4f361bd (cached memory) | 79 | 73 | 76 | 73 |
| baseline, same run | 468 | 468 | 460 | 452 |
| check.py (4f361bd), 9 runs | 77 | 80 | 80 | 78 |
| I/O floor (read n, print length) | 8 | 8 | 8 | 6 |

Medians in ms; 7 rounds (first two rows) or 9 rounds (final). Time limit 2000 ms.

- Phase profile of the first try (max2): read 4 ms; one 3^(k-2) by squaring 40 ms (multiply 27, carry 9);
  the fix-up loop and comparisons about 50 ms (10 scalar multiplies over 1.5·10^6 digits). Hence the
  log3 estimate: 3 scalar multiplies instead of about 10.
- Result: ours 73-79 ms, baseline 452-468 ms (6.1x). Both pass 2 s: on this problem a textbook NTT with
  one digit per coefficient passes too, 4.3x under the limit. The gain is a ratio, not pass/fail.
- check.py (gcc:14): samples OK, Windows build (mingw-w64, Wine) samples OK, stress 500 cases OK
  (and 300 more after the rebundle).
  Windows build under Wine on all 4 max cases: correct answers, exit 0 (max2: 0.13 s under Wine).
- ASan + UBSan (gcc:14, -O1): 60 small cases and max2, clean.
- Answers: max0 9431565, max1 9431559, max2 9431562 (= 3k), max3 9431563.
- Caveat: gcc:14 without -march warns "AVX vector return without AVX enabled changes the ABI" (-Wpsabi)
  inside lib/poly. Guess: harmless (an internal lambda; stress and max cases pass). It shows in the compile log.
