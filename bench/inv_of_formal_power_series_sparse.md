# inv_of_formal_power_series_sparse

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 6.9.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 4.5.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small_dense_00 | `N K = 1000000 5` | 70 B | 9.4 MB | 6.8 | 4.8 | 1.9 | 4.3 | 3.9 | 0.4 |
| small_dense_02 | `N K = 1000000 8` | 104 B | 9.4 MB | 6.8 | 4.7 | 2.1 | 4.4 | 3.6 | 0.8 |
| small_dense_01 | `N K = 1000000 6` | 81 B | 9.4 MB | 6.7 | 4.8 | 2.0 | 4.3 | 3.9 | 0.5 |
| small_dense_04 | `N K = 1000000 4` | 58 B | 9.4 MB | 6.5 | 4.5 | 2.0 | 4.2 | 3.7 | 0.5 |
| max_random_00 | `N K = 1000000 10` | 175 B | 1.9 MB | 3.3 | 1.7 | 1.6 | 2.3 | 1.0 | 1.3 |
| max_random_01 | `N K = 1000000 10` | 173 B | 1.9 MB | 3.2 | 1.6 | 1.6 | 2.3 | 1.0 | 1.3 |
| max_random_02 | `N K = 1000000 10` | 174 B | 1.9 MB | 3.1 | 1.4 | 1.7 | 2.2 | 1.1 | 1.1 |
| max_random_03 | `N K = 1000000 10` | 175 B | 1.9 MB | 3.1 | 1.5 | 1.6 | 2.2 | 1.0 | 1.2 |
| max_random_04 | `N K = 1000000 10` | 174 B | 1.9 MB | 3.0 | 1.4 | 1.6 | 2.1 | 1.0 | 1.1 |
| random_04 | `N K = 801300 3` | 55 B | 1.5 MB | 2.2 | 1.3 | 0.9 | 1.5 | 0.9 | 0.6 |
| small_dense_03 | `N K = 1000000 1` | 22 B | 1.9 MB | 2.2 | 1.5 | 0.8 | 1.4 | 1.0 | 0.4 |
| random_02 | `N K = 577624 6` | 105 B | 1.1 MB | 2.1 | 1.2 | 1.0 | 1.5 | 0.8 | 0.7 |
| random_01 | `N K = 463046 5` | 89 B | 904.4 KB | 1.9 | 1.1 | 0.8 | 1.2 | 0.7 | 0.5 |
| random_03 | `N K = 429249 2` | 38 B | 838.4 KB | 1.7 | 1.1 | 0.7 | 1.2 | 0.7 | 0.5 |
| min_K_01 | `N K = 463046 1` | 21 B | 904.4 KB | 1.7 | 1.1 | 0.6 | 1.1 | 0.7 | 0.4 |
| random_00 | `N K = 389813 1` | 21 B | 761.4 KB | 1.6 | 1.0 | 0.6 | 1.0 | 0.7 | 0.3 |
| min_K_00 | `N K = 389813 1` | 21 B | 761.4 KB | 1.6 | 1.1 | 0.5 | 1.0 | 0.7 | 0.3 |
| example_00 | `N K = 5 2` | 12 B | 18 B | 1.3 | 0.7 | 0.6 | 0.8 | 0.4 | 0.3 |
| example_01 | `N K = 10 5` | 25 B | 100 B | 1.3 | 0.7 | 0.5 | 0.7 | 0.4 | 0.3 |
| small_N_03 | `N K = 1 1` | 16 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_N_01 | `N K = 6 3` | 39 B | 35 B | 1.1 | 0.8 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_N_00 | `N K = 5 3` | 40 B | 50 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_N_04 | `N K = 4 1` | 16 B | 16 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_N_02 | `N K = 8 3` | 40 B | 48 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
