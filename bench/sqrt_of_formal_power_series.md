# sqrt_of_formal_power_series

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.1. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 11.8. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_random_01 | `N = 500000` | 4.7 MB | 4.7 MB | 12.1 | 3.8 | 8.3 | 11.6 | 2.5 | 9.1 |
| random_01 | `N = 500000` | 4.7 MB | 4.7 MB | 12.0 | 3.8 | 8.2 | 11.6 | 2.6 | 9.0 |
| max_random_02 | `N = 500000` | 4.7 MB | 4.7 MB | 12.0 | 3.8 | 8.2 | 11.5 | 2.5 | 9.0 |
| random_02 | `N = 500000` | 4.7 MB | 4.7 MB | 11.9 | 3.8 | 8.1 | 11.6 | 2.5 | 9.1 |
| monomial_02 | `N = 500000` | 976.6 KB | 976.6 KB | 11.5 | 1.3 | 10.2 | 11.3 | 0.8 | 10.4 |
| lower_deg_zero_00 | `N = 500000` | 3.6 MB | 4.1 MB | 10.5 | 3.4 | 7.1 | 9.8 | 2.3 | 7.5 |
| near_262144_02 | `N = 262145` | 2.5 MB | 2.5 MB | 8.6 | 2.3 | 6.2 | 8.5 | 1.5 | 7.0 |
| monomial_00 | `N = 500000` | 976.6 KB | 976.6 KB | 7.9 | 1.3 | 6.6 | 6.9 | 0.8 | 6.1 |
| lower_deg_zero_01 | `N = 500000` | 2.1 MB | 3.4 MB | 7.4 | 2.8 | 4.7 | 6.2 | 1.8 | 4.4 |
| near_262144_01 | `N = 262144` | 2.5 MB | 2.5 MB | 6.4 | 2.3 | 4.1 | 5.9 | 1.5 | 4.4 |
| all_zero_01 | `N = 463046` | 904.4 KB | 904.4 KB | 4.5 | 1.3 | 3.2 | 2.9 | 0.8 | 2.1 |
| all_zero_00 | `N = 389813` | 761.4 KB | 761.4 KB | 4.1 | 1.2 | 2.9 | 2.6 | 0.7 | 1.9 |
| lower_deg_zero_03 | `N = 500000` | 4.0 MB | 3 B | 2.3 | 1.3 | 1.0 | 1.6 | 0.9 | 0.7 |
| lower_deg_zero_05 | `N = 500000` | 2.9 MB | 3 B | 2.2 | 1.1 | 1.1 | 1.6 | 0.7 | 0.9 |
| max_random_00 | `N = 500000` | 4.7 MB | 3 B | 2.2 | 1.4 | 0.8 | 1.5 | 0.9 | 0.6 |
| lower_deg_zero_06 | `N = 500000` | 2.5 MB | 3 B | 2.2 | 1.1 | 1.1 | 1.6 | 0.7 | 0.9 |
| lower_deg_zero_02 | `N = 500000` | 2.7 MB | 3 B | 2.1 | 1.1 | 1.0 | 1.5 | 0.7 | 0.8 |
| random_00 | `N = 500000` | 4.7 MB | 3 B | 2.1 | 1.4 | 0.8 | 1.5 | 0.9 | 0.7 |
| lower_deg_zero_07 | `N = 500000` | 2.0 MB | 3 B | 2.1 | 1.0 | 1.1 | 1.5 | 0.6 | 0.9 |
| lower_deg_zero_04 | `N = 500000` | 1.3 MB | 3 B | 2.0 | 0.9 | 1.1 | 1.4 | 0.6 | 0.8 |
| monomial_01 | `N = 500000` | 976.6 KB | 3 B | 1.9 | 0.8 | 1.1 | 1.4 | 0.5 | 0.9 |
| monomial_03 | `N = 500000` | 976.6 KB | 3 B | 1.9 | 0.8 | 1.1 | 1.4 | 0.5 | 0.8 |
| near_262144_00 | `N = 262143` | 2.5 MB | 3 B | 1.7 | 1.0 | 0.6 | 1.2 | 0.7 | 0.4 |
| example_00 | `N = 4` | 11 B | 16 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| example_01 | `N = 4` | 12 B | 3 B | 1.1 | 0.8 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_03 | `N = 4` | 42 B | 39 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_00 | `N = 1` | 12 B | 3 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_06 | `N = 7` | 71 B | 3 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_05 | `N = 6` | 62 B | 58 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_01 | `N = 2` | 22 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_02 | `N = 3` | 30 B | 29 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_04 | `N = 5` | 52 B | 49 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_09 | `N = 10` | 103 B | 3 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_degree_07 | `N = 8` | 82 B | 3 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_08 | `N = 9` | 89 B | 90 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
