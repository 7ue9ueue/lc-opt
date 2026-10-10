# log_of_formal_power_series

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 14.4. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 14.3. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_random_00 | `N = 500000` | 4.7 MB | 4.7 MB | 14.3 | 3.8 | 10.5 | 14.1 | 2.5 | 11.6 |
| max_random_01 | `N = 500000` | 4.7 MB | 4.7 MB | 14.3 | 3.8 | 10.5 | 14.0 | 2.5 | 11.5 |
| random_01 | `N = 463046` | 4.4 MB | 4.4 MB | 14.1 | 3.5 | 10.6 | 14.0 | 2.4 | 11.6 |
| max_random_02 | `N = 500000` | 4.7 MB | 4.7 MB | 14.1 | 3.8 | 10.4 | 14.2 | 2.6 | 11.6 |
| max_random_03 | `N = 500000` | 4.7 MB | 4.7 MB | 14.1 | 3.8 | 10.3 | 14.3 | 2.7 | 11.6 |
| max_random_04 | `N = 500000` | 4.7 MB | 4.7 MB | 14.1 | 3.7 | 10.3 | 14.3 | 2.6 | 11.6 |
| max_all_zero_00 | `N = 500000` | 976.6 KB | 976.6 KB | 13.9 | 1.3 | 12.6 | 14.2 | 0.8 | 13.4 |
| random_03 | `N = 429249` | 4.0 MB | 4.0 MB | 13.7 | 3.4 | 10.3 | 13.9 | 2.3 | 11.6 |
| random_00 | `N = 389813` | 3.7 MB | 3.7 MB | 11.1 | 3.1 | 8.0 | 10.8 | 2.1 | 8.7 |
| random_04 | `N = 277012` | 2.6 MB | 2.6 MB | 10.0 | 2.4 | 7.6 | 10.1 | 1.6 | 8.6 |
| near_262144_02 | `N = 262145` | 2.5 MB | 2.5 MB | 7.6 | 2.3 | 5.3 | 7.3 | 1.5 | 5.8 |
| near_262144_00 | `N = 262143` | 2.5 MB | 2.5 MB | 7.5 | 2.3 | 5.2 | 7.3 | 1.6 | 5.7 |
| near_262144_01 | `N = 262144` | 2.5 MB | 2.5 MB | 7.5 | 2.3 | 5.1 | 7.3 | 1.6 | 5.8 |
| random_02 | `N = 53336` | 515.1 KB | 515.0 KB | 2.6 | 1.0 | 1.5 | 2.1 | 0.7 | 1.5 |
| example_00 | `N = 5` | 36 B | 10 B | 1.3 | 0.7 | 0.6 | 0.7 | 0.5 | 0.3 |
| small_degree_08 | `N = 9` | 81 B | 82 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_00 | `N = 1` | 4 B | 2 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_06 | `N = 7` | 63 B | 62 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_01 | `N = 2` | 14 B | 12 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_05 | `N = 6` | 54 B | 50 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_02 | `N = 3` | 24 B | 22 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_03 | `N = 4` | 34 B | 32 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_04 | `N = 5` | 44 B | 42 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_09 | `N = 10` | 95 B | 90 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_07 | `N = 8` | 74 B | 71 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
