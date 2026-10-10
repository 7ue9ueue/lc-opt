# exp_of_formal_power_series_sparse

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 8.7. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 6.6. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small_dense_02 | `N K = 1000000 7` | 92 B | 9.4 MB | 8.7 | 4.6 | 4.1 | 6.5 | 3.5 | 3.0 |
| small_dense_01 | `N K = 1000000 5` | 69 B | 9.4 MB | 8.7 | 4.4 | 4.2 | 6.4 | 3.7 | 2.7 |
| small_dense_00 | `N K = 1000000 4` | 58 B | 9.4 MB | 8.5 | 4.6 | 3.9 | 6.4 | 3.9 | 2.5 |
| small_dense_04 | `N K = 1000000 3` | 46 B | 9.4 MB | 8.5 | 4.6 | 3.8 | 6.4 | 3.5 | 2.9 |
| max_random_01 | `N K = 1000000 10` | 178 B | 1.9 MB | 3.9 | 1.6 | 2.3 | 2.9 | 1.1 | 1.8 |
| max_random_00 | `N K = 1000000 10` | 180 B | 1.9 MB | 3.8 | 1.7 | 2.2 | 3.0 | 1.0 | 1.9 |
| max_random_02 | `N K = 1000000 10` | 179 B | 1.9 MB | 3.7 | 1.5 | 2.3 | 2.7 | 1.0 | 1.7 |
| max_random_03 | `N K = 1000000 10` | 180 B | 1.9 MB | 3.7 | 1.4 | 2.3 | 2.8 | 1.1 | 1.7 |
| max_random_04 | `N K = 1000000 10` | 179 B | 1.9 MB | 3.6 | 1.5 | 2.2 | 2.7 | 1.1 | 1.6 |
| random_02 | `N K = 577624 10` | 175 B | 1.1 MB | 2.7 | 1.2 | 1.5 | 2.0 | 0.8 | 1.2 |
| random_04 | `N K = 801300 2` | 43 B | 1.5 MB | 2.7 | 1.3 | 1.4 | 1.9 | 0.9 | 1.0 |
| random_01 | `N K = 463046 10` | 175 B | 911.8 KB | 2.5 | 1.2 | 1.3 | 1.8 | 0.7 | 1.1 |
| small_dense_03 | `N K = 1000000 0` | 10 B | 1.9 MB | 2.3 | 1.4 | 0.8 | 1.5 | 1.0 | 0.4 |
| random_00 | `N K = 389813 10` | 178 B | 762.2 KB | 2.2 | 1.1 | 1.1 | 1.6 | 0.7 | 0.9 |
| random_03 | `N K = 429249 1` | 26 B | 838.4 KB | 2.0 | 1.0 | 1.0 | 1.4 | 0.7 | 0.7 |
| min_K_01 | `N K = 463046 0` | 9 B | 904.4 KB | 1.7 | 1.0 | 0.7 | 1.1 | 0.7 | 0.4 |
| min_K_00 | `N K = 389813 0` | 9 B | 761.4 KB | 1.6 | 1.0 | 0.6 | 1.0 | 0.7 | 0.4 |
| example_00 | `N K = 5 1` | 8 B | 18 B | 1.4 | 0.7 | 0.6 | 0.8 | 0.4 | 0.4 |
| example_01 | `N K = 10 4` | 21 B | 84 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| small_N_03 | `N K = 1 0` | 4 B | 2 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| small_N_02 | `N K = 8 2` | 28 B | 32 B | 1.2 | 0.8 | 0.4 | 0.8 | 0.5 | 0.3 |
| example_02 | `N K = 10 0` | 5 B | 20 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| small_N_04 | `N K = 4 0` | 4 B | 8 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.4 | 0.3 |
| small_N_01 | `N K = 6 2` | 27 B | 27 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.4 |
| small_N_00 | `N K = 5 2` | 28 B | 41 B | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
