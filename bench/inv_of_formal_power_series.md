# inv_of_formal_power_series

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.0. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.0. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_random_00 | `N = 500000` | 4.7 MB | 4.7 MB | 12.0 | 3.8 | 8.2 | 12.0 | 2.5 | 9.5 |
| max_random_03 | `N = 500000` | 4.7 MB | 4.7 MB | 12.0 | 3.8 | 8.1 | 11.8 | 2.6 | 9.2 |
| max_random_01 | `N = 500000` | 4.7 MB | 4.7 MB | 12.0 | 3.9 | 8.0 | 11.6 | 2.6 | 9.0 |
| max_random_02 | `N = 500000` | 4.7 MB | 4.7 MB | 11.9 | 3.8 | 8.1 | 11.6 | 2.5 | 9.0 |
| max_random_or_zero_00 | `N = 500000` | 4.7 MB | 4.7 MB | 11.9 | 3.8 | 8.1 | 11.9 | 2.5 | 9.4 |
| max_random_04 | `N = 500000` | 4.7 MB | 4.7 MB | 11.9 | 3.8 | 8.1 | 11.8 | 2.5 | 9.3 |
| random_01 | `N = 463046` | 4.4 MB | 4.4 MB | 11.7 | 3.6 | 8.2 | 11.6 | 2.3 | 9.3 |
| random_03 | `N = 429249` | 4.0 MB | 4.0 MB | 11.4 | 3.3 | 8.1 | 11.5 | 2.2 | 9.3 |
| random_00 | `N = 389813` | 3.7 MB | 3.7 MB | 11.2 | 3.1 | 8.1 | 11.3 | 2.1 | 9.3 |
| random_04 | `N = 277012` | 2.6 MB | 2.6 MB | 10.3 | 2.3 | 8.0 | 10.7 | 1.6 | 9.1 |
| near_262144_02 | `N = 262145` | 2.5 MB | 2.5 MB | 10.3 | 2.3 | 7.9 | 10.7 | 1.5 | 9.1 |
| near_262144_00 | `N = 262143` | 2.5 MB | 2.5 MB | 6.5 | 2.3 | 4.1 | 6.0 | 1.5 | 4.5 |
| near_262144_01 | `N = 262144` | 2.5 MB | 2.5 MB | 6.4 | 2.3 | 4.2 | 6.0 | 1.5 | 4.5 |
| random_02 | `N = 53336` | 515.1 KB | 515.1 KB | 2.3 | 1.0 | 1.2 | 1.9 | 0.7 | 1.2 |
| example_00 | `N = 5` | 12 B | 50 B | 1.3 | 0.7 | 0.6 | 0.7 | 0.5 | 0.2 |
| small_degree_09 | `N = 10` | 103 B | 100 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_06 | `N = 7` | 71 B | 69 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_00 | `N = 1` | 12 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_03 | `N = 4` | 42 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_degree_05 | `N = 6` | 62 B | 59 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_01 | `N = 2` | 22 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_02 | `N = 3` | 30 B | 30 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_degree_08 | `N = 9` | 89 B | 88 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_04 | `N = 5` | 52 B | 49 B | 1.1 | 0.7 | 0.3 | 0.7 | 0.5 | 0.2 |
| small_degree_07 | `N = 8` | 82 B | 80 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
