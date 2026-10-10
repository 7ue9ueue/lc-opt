# log_of_formal_power_series_sparse

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 7.3. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 5.1. Older `main.cpp` than the current one.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| small_dense_01 | 73 B | 9.4 MB | `1000000 6` | 7.3 | 4.5 | 2.7 | 4.9 | 3.7 | 1.2 |
| small_dense_00 | 62 B | 9.4 MB | `1000000 5` | 7.2 | 5.5 | 1.7 | 4.8 | 3.8 | 1.0 |
| small_dense_02 | 96 B | 9.4 MB | `1000000 8` | 7.2 | 4.7 | 2.6 | 5.0 | 3.5 | 1.5 |
| small_dense_04 | 50 B | 9.4 MB | `1000000 4` | 7.1 | 4.4 | 2.7 | 4.8 | 3.5 | 1.4 |
| max_random_00 | 167 B | 1.9 MB | `1000000 10` | 3.8 | 1.7 | 2.1 | 2.9 | 1.0 | 1.9 |
| max_random_01 | 165 B | 1.9 MB | `1000000 10` | 3.8 | 1.6 | 2.1 | 2.9 | 1.1 | 1.9 |
| max_random_02 | 166 B | 1.9 MB | `1000000 10` | 3.7 | 1.5 | 2.2 | 2.8 | 1.0 | 1.7 |
| max_random_03 | 167 B | 1.9 MB | `1000000 10` | 3.6 | 1.4 | 2.2 | 2.8 | 1.0 | 1.8 |
| max_random_04 | 166 B | 1.9 MB | `1000000 10` | 3.6 | 1.4 | 2.1 | 2.8 | 1.0 | 1.8 |
| small_dense_03 | 14 B | 1.9 MB | `1000000 1` | 2.8 | 1.5 | 1.3 | 2.0 | 1.0 | 0.9 |
| random_04 | 47 B | 1.5 MB | `801300 3` | 2.7 | 1.3 | 1.4 | 2.0 | 0.9 | 1.0 |
| random_02 | 97 B | 1.1 MB | `577624 6` | 2.5 | 1.2 | 1.3 | 1.9 | 0.8 | 1.1 |
| random_01 | 81 B | 904.4 KB | `463046 5` | 2.2 | 1.0 | 1.1 | 1.6 | 0.7 | 0.9 |
| random_03 | 30 B | 838.4 KB | `429249 2` | 2.1 | 1.0 | 1.0 | 1.5 | 0.7 | 0.8 |
| min_K_01 | 13 B | 904.4 KB | `463046 1` | 2.0 | 1.0 | 0.9 | 1.4 | 0.7 | 0.7 |
| min_K_00 | 13 B | 761.4 KB | `389813 1` | 1.9 | 1.0 | 0.9 | 1.3 | 0.6 | 0.7 |
| random_00 | 13 B | 761.4 KB | `389813 1` | 1.9 | 1.0 | 0.9 | 1.3 | 0.6 | 0.7 |
| example_00 | 12 B | 18 B | `5 2` | 1.4 | 0.7 | 0.7 | 0.8 | 0.5 | 0.4 |
| example_01 | 49 B | 60 B | `10 5` | 1.2 | 0.7 | 0.5 | 0.8 | 0.4 | 0.4 |
| small_N_01 | 32 B | 28 B | `6 3` | 1.2 | 0.7 | 0.5 | 0.8 | 0.4 | 0.4 |
| small_N_00 | 32 B | 41 B | `5 3` | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.4 |
| small_N_04 | 8 B | 8 B | `4 1` | 1.2 | 0.7 | 0.5 | 0.8 | 0.4 | 0.4 |
| small_N_02 | 32 B | 40 B | `8 3` | 1.2 | 0.7 | 0.5 | 0.9 | 0.4 | 0.4 |
| small_N_03 | 8 B | 2 B | `1 1` | 1.2 | 0.7 | 0.4 | 0.8 | 0.4 | 0.3 |
