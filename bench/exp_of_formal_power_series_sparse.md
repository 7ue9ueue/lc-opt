# exp_of_formal_power_series_sparse

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 8.7. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 6.6. Older `main.cpp` than the current one.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| small_dense_02 | 92 B | 9.4 MB | `1000000 7` | 8.7 | 4.6 | 4.1 | 6.5 | 3.5 | 3.0 |
| small_dense_01 | 69 B | 9.4 MB | `1000000 5` | 8.7 | 4.4 | 4.2 | 6.4 | 3.7 | 2.7 |
| small_dense_00 | 58 B | 9.4 MB | `1000000 4` | 8.5 | 4.6 | 3.9 | 6.4 | 3.9 | 2.5 |
| small_dense_04 | 46 B | 9.4 MB | `1000000 3` | 8.5 | 4.6 | 3.8 | 6.4 | 3.5 | 2.9 |
| max_random_01 | 178 B | 1.9 MB | `1000000 10` | 3.9 | 1.6 | 2.3 | 2.9 | 1.1 | 1.8 |
| max_random_00 | 180 B | 1.9 MB | `1000000 10` | 3.8 | 1.7 | 2.2 | 3.0 | 1.0 | 1.9 |
| max_random_02 | 179 B | 1.9 MB | `1000000 10` | 3.7 | 1.5 | 2.3 | 2.7 | 1.0 | 1.7 |
| max_random_03 | 180 B | 1.9 MB | `1000000 10` | 3.7 | 1.4 | 2.3 | 2.8 | 1.1 | 1.7 |
| max_random_04 | 179 B | 1.9 MB | `1000000 10` | 3.6 | 1.5 | 2.2 | 2.7 | 1.1 | 1.6 |
| random_02 | 175 B | 1.1 MB | `577624 10` | 2.7 | 1.2 | 1.5 | 2.0 | 0.8 | 1.2 |
| random_04 | 43 B | 1.5 MB | `801300 2` | 2.7 | 1.3 | 1.4 | 1.9 | 0.9 | 1.0 |
| random_01 | 175 B | 911.8 KB | `463046 10` | 2.5 | 1.2 | 1.3 | 1.8 | 0.7 | 1.1 |
| small_dense_03 | 10 B | 1.9 MB | `1000000 0` | 2.3 | 1.4 | 0.8 | 1.5 | 1.0 | 0.4 |
| random_00 | 178 B | 762.2 KB | `389813 10` | 2.2 | 1.1 | 1.1 | 1.6 | 0.7 | 0.9 |
| random_03 | 26 B | 838.4 KB | `429249 1` | 2.0 | 1.0 | 1.0 | 1.4 | 0.7 | 0.7 |
| min_K_01 | 9 B | 904.4 KB | `463046 0` | 1.7 | 1.0 | 0.7 | 1.1 | 0.7 | 0.4 |
| min_K_00 | 9 B | 761.4 KB | `389813 0` | 1.6 | 1.0 | 0.6 | 1.0 | 0.7 | 0.4 |
| example_00 | 8 B | 18 B | `5 1` | 1.4 | 0.7 | 0.6 | 0.8 | 0.4 | 0.4 |
| example_01 | 21 B | 84 B | `10 4` | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| small_N_03 | 4 B | 2 B | `1 0` | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| small_N_02 | 28 B | 32 B | `8 2` | 1.2 | 0.8 | 0.4 | 0.8 | 0.5 | 0.3 |
| example_02 | 5 B | 20 B | `10 0` | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| small_N_04 | 4 B | 8 B | `4 0` | 1.2 | 0.7 | 0.5 | 0.8 | 0.4 | 0.3 |
| small_N_01 | 27 B | 27 B | `6 2` | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.4 |
| small_N_00 | 28 B | 41 B | `5 2` | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
