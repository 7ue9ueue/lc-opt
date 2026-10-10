# log_of_formal_power_series_sparse

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 7.3. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 5.1. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small_dense_01 | `N K = 1000000 6` | 73 B | 9.4 MB | 7.3 | 4.5 | 2.7 | 4.9 | 3.7 | 1.2 |
| small_dense_00 | `N K = 1000000 5` | 62 B | 9.4 MB | 7.2 | 5.5 | 1.7 | 4.8 | 3.8 | 1.0 |
| small_dense_02 | `N K = 1000000 8` | 96 B | 9.4 MB | 7.2 | 4.7 | 2.6 | 5.0 | 3.5 | 1.5 |
| small_dense_04 | `N K = 1000000 4` | 50 B | 9.4 MB | 7.1 | 4.4 | 2.7 | 4.8 | 3.5 | 1.4 |
| max_random_00 | `N K = 1000000 10` | 167 B | 1.9 MB | 3.8 | 1.7 | 2.1 | 2.9 | 1.0 | 1.9 |
| max_random_01 | `N K = 1000000 10` | 165 B | 1.9 MB | 3.8 | 1.6 | 2.1 | 2.9 | 1.1 | 1.9 |
| max_random_02 | `N K = 1000000 10` | 166 B | 1.9 MB | 3.7 | 1.5 | 2.2 | 2.8 | 1.0 | 1.7 |
| max_random_03 | `N K = 1000000 10` | 167 B | 1.9 MB | 3.6 | 1.4 | 2.2 | 2.8 | 1.0 | 1.8 |
| max_random_04 | `N K = 1000000 10` | 166 B | 1.9 MB | 3.6 | 1.4 | 2.1 | 2.8 | 1.0 | 1.8 |
| small_dense_03 | `N K = 1000000 1` | 14 B | 1.9 MB | 2.8 | 1.5 | 1.3 | 2.0 | 1.0 | 0.9 |
| random_04 | `N K = 801300 3` | 47 B | 1.5 MB | 2.7 | 1.3 | 1.4 | 2.0 | 0.9 | 1.0 |
| random_02 | `N K = 577624 6` | 97 B | 1.1 MB | 2.5 | 1.2 | 1.3 | 1.9 | 0.8 | 1.1 |
| random_01 | `N K = 463046 5` | 81 B | 904.4 KB | 2.2 | 1.0 | 1.1 | 1.6 | 0.7 | 0.9 |
| random_03 | `N K = 429249 2` | 30 B | 838.4 KB | 2.1 | 1.0 | 1.0 | 1.5 | 0.7 | 0.8 |
| min_K_01 | `N K = 463046 1` | 13 B | 904.4 KB | 2.0 | 1.0 | 0.9 | 1.4 | 0.7 | 0.7 |
| min_K_00 | `N K = 389813 1` | 13 B | 761.4 KB | 1.9 | 1.0 | 0.9 | 1.3 | 0.6 | 0.7 |
| random_00 | `N K = 389813 1` | 13 B | 761.4 KB | 1.9 | 1.0 | 0.9 | 1.3 | 0.6 | 0.7 |
| example_00 | `N K = 5 2` | 12 B | 18 B | 1.4 | 0.7 | 0.7 | 0.8 | 0.5 | 0.4 |
| example_01 | `N K = 10 5` | 49 B | 60 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.4 | 0.4 |
| small_N_01 | `N K = 6 3` | 32 B | 28 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.4 | 0.4 |
| small_N_00 | `N K = 5 3` | 32 B | 41 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.4 |
| small_N_04 | `N K = 4 1` | 8 B | 8 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.4 | 0.4 |
| small_N_02 | `N K = 8 3` | 32 B | 40 B | 1.2 | 0.7 | 0.5 | 0.9 | 0.4 | 0.4 |
| small_N_03 | `N K = 1 1` | 8 B | 2 B | 1.2 | 0.7 | 0.4 | 0.8 | 0.4 | 0.3 |
