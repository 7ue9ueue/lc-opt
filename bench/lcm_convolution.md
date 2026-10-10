# lcm_convolution

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 16.2.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 14.0.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| near_prime_squared_00 | `N = 994008` | 18.7 MB | 9.4 MB | 16.1 | 8.0 | 8.2 | 13.9 | 6.3 | 7.6 |
| near_prime_02 | `N = 999984` | 18.9 MB | 9.4 MB | 15.5 | 7.5 | 8.1 | 13.5 | 6.0 | 7.5 |
| max_random_00 | `N = 1000000` | 18.9 MB | 9.4 MB | 15.5 | 7.5 | 8.0 | 13.7 | 5.7 | 8.0 |
| near_prime_squared_01 | `N = 994009` | 18.7 MB | 9.4 MB | 15.5 | 8.6 | 6.9 | 13.3 | 6.0 | 7.3 |
| max_random_01 | `N = 1000000` | 18.9 MB | 9.4 MB | 15.5 | 8.3 | 7.2 | 13.7 | 5.9 | 7.8 |
| near_prime_squared_02 | `N = 994010` | 18.7 MB | 9.4 MB | 15.5 | 8.5 | 7.0 | 13.3 | 6.1 | 7.2 |
| near_prime_00 | `N = 999982` | 18.9 MB | 9.4 MB | 15.4 | 7.6 | 7.8 | 13.8 | 6.1 | 7.7 |
| near_prime_01 | `N = 999983` | 18.9 MB | 9.4 MB | 15.4 | 7.9 | 7.5 | 13.6 | 5.9 | 7.6 |
| random_02 | `N = 577624` | 10.9 MB | 5.4 MB | 9.7 | 4.6 | 5.1 | 7.8 | 3.7 | 4.1 |
| random_01 | `N = 463046` | 8.7 MB | 4.4 MB | 7.7 | 3.9 | 3.8 | 6.2 | 3.1 | 3.1 |
| random_00 | `N = 389813` | 7.4 MB | 3.7 MB | 6.8 | 3.6 | 3.2 | 5.4 | 2.6 | 2.7 |
| all_zero_00 | `N = 6` | 26 B | 12 B | 1.3 | 0.7 | 0.6 | 0.8 | 0.5 | 0.4 |
| small_04 | `N = 5` | 102 B | 50 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_00 | `N = 1` | 22 B | 10 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_01 | `N = 2` | 41 B | 20 B | 1.2 | 0.8 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_02 | `N = 3` | 60 B | 28 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_07 | `N = 8` | 159 B | 78 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_15 | `N = 16` | 320 B | 159 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| example_00 | `N = 6` | 26 B | 18 B | 1.1 | 0.7 | 0.4 | 0.8 | 0.4 | 0.3 |
| small_06 | `N = 7` | 140 B | 70 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_03 | `N = 4` | 82 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_08 | `N = 9` | 179 B | 90 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_12 | `N = 13` | 261 B | 128 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_09 | `N = 10` | 203 B | 98 B | 1.1 | 0.8 | 0.3 | 0.7 | 0.5 | 0.2 |
| small_05 | `N = 6` | 121 B | 60 B | 1.1 | 0.8 | 0.3 | 0.7 | 0.5 | 0.2 |
| small_11 | `N = 12` | 240 B | 119 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_10 | `N = 11` | 222 B | 110 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_14 | `N = 15` | 301 B | 149 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_13 | `N = 14` | 283 B | 138 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
