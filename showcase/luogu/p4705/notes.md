# Luogu P4705 玩游戏

https://www.luogu.com.cn/problem/P4705 (洛谷 2018 年 5 月月赛, LGR-047). 3 s, 512 MB.

## Problem

n, m <= 10^5 values a_i, b_j in [0, P), P = 998244353; t <= 10^5. For k = 1..t print
E[(a_x + b_y)^k] over uniform x, y, mod P.

## Solutions

- Answer k = k!/(nm) sum_j (S_a(j)/j!) (S_b(k-j)/(k-j)!), S_a(j) = sum_i a_i^j: one product.
- Power sums: S_a(j) = -j [x^j] log prod_i (1 - a_i x), log in O(t log t). The product of n linear
  factors is the costly step; the standard route builds it by divide and conquer, O(n log^2 n).
  An equivalent route sums 1/(1 - a_i x) as fractions by divide and conquer (same order, more products).
- Ours: prod_i (1 - a_i x) by divide and conquer (lib/easy multiply), then `easy::log` mod x^(t+1),
  then the product above. O(n log^2 n + t log t). Same route in `baseline.cpp` on a textbook NTT
  (iterative radix-2, bit reversal, `% P`, roots by repeated multiplication, Newton inverse for log).

## Evidence the route is tight

Forum thread titles on the problem page (contents need login):
- 314279, 2021-05-03: "萌新求助，分治乘法就 TLE 了" (divide-and-conquer multiplication already TLEs).
- 209651, 2020-03-31: "请求开大时限" (asking for a larger time limit).
- 392872 "萌新求助分治 NTT", 345868 "关于NTT分治乘".
Guess: the limit was raised to the current 3 s at some point after 209651; not confirmed.

## Measurements

lc-amd (EPYC 7B13), gcc:13 image, `-std=c++20 -O2`, max cases n = m = t = 10^5 (gen.py max, seeds 0, 1).
Timed inside `flock /tmp/bench.lock`, 7 runs each, 2026-10-10. The VM was loaded (load average 4.4
on 4 vCPUs), so spread is wide.

| Program | seed 0 (ms) | seed 1 (ms) |
|---|---|---|
| main.cpp | 66-136, median 74 (an earlier quieter run: median 42) | 65-115, median 71 |
| baseline.cpp | 922-1027, median 1018 | 889-1044, median 993 |

Ratio about 14x (25x on the quiet run). Both outputs identical on both max cases.
The baseline also fits in 3 s on lc-amd: at the current limit this problem shows a gap, not a
pass/fail split. Under a 1 s limit (guess: the original) the baseline fails and ours passes.

## Checks

- Samples 1, 2: OK. Stress: 300 cases against brute.cpp (O(n m t)), sizes up to 300/300/400 so
  lib/easy's NTT paths run. ASan/UBSan (gcc 13, -O1): 60 small cases and max seed 0, clean.
- GCC 13 `-Wall` on main.cpp warns inside lib (not this file): `-Warray-bounds` in a
  `std::array<uint32_t, 21>` access, `-Winvalid-constexpr` on `table_words`, `-Wpsabi`.
