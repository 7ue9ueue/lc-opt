# inv_of_formal_power_series_sparse

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 6.9.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 4.5.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| small_dense_00 | 70 B | 9.4 MB | `1000000 5` | 6.8 | 4.8 | 1.9 | 4.3 | 3.9 | 0.4 |
| small_dense_02 | 104 B | 9.4 MB | `1000000 8` | 6.8 | 4.7 | 2.1 | 4.4 | 3.6 | 0.8 |
| small_dense_01 | 81 B | 9.4 MB | `1000000 6` | 6.7 | 4.8 | 2.0 | 4.3 | 3.9 | 0.5 |
| small_dense_04 | 58 B | 9.4 MB | `1000000 4` | 6.5 | 4.5 | 2.0 | 4.2 | 3.7 | 0.5 |
| max_random_00 | 175 B | 1.9 MB | `1000000 10` | 3.3 | 1.7 | 1.6 | 2.3 | 1.0 | 1.3 |
| max_random_01 | 173 B | 1.9 MB | `1000000 10` | 3.2 | 1.6 | 1.6 | 2.3 | 1.0 | 1.3 |
| max_random_02 | 174 B | 1.9 MB | `1000000 10` | 3.1 | 1.4 | 1.7 | 2.2 | 1.1 | 1.1 |
| max_random_03 | 175 B | 1.9 MB | `1000000 10` | 3.1 | 1.5 | 1.6 | 2.2 | 1.0 | 1.2 |
| max_random_04 | 174 B | 1.9 MB | `1000000 10` | 3.0 | 1.4 | 1.6 | 2.1 | 1.0 | 1.1 |
| random_04 | 55 B | 1.5 MB | `801300 3` | 2.2 | 1.3 | 0.9 | 1.5 | 0.9 | 0.6 |
| small_dense_03 | 22 B | 1.9 MB | `1000000 1` | 2.2 | 1.5 | 0.8 | 1.4 | 1.0 | 0.4 |
| random_02 | 105 B | 1.1 MB | `577624 6` | 2.1 | 1.2 | 1.0 | 1.5 | 0.8 | 0.7 |
| random_01 | 89 B | 904.4 KB | `463046 5` | 1.9 | 1.1 | 0.8 | 1.2 | 0.7 | 0.5 |
| random_03 | 38 B | 838.4 KB | `429249 2` | 1.7 | 1.1 | 0.7 | 1.2 | 0.7 | 0.5 |
| min_K_01 | 21 B | 904.4 KB | `463046 1` | 1.7 | 1.1 | 0.6 | 1.1 | 0.7 | 0.4 |
| random_00 | 21 B | 761.4 KB | `389813 1` | 1.6 | 1.0 | 0.6 | 1.0 | 0.7 | 0.3 |
| min_K_00 | 21 B | 761.4 KB | `389813 1` | 1.6 | 1.1 | 0.5 | 1.0 | 0.7 | 0.3 |
| example_00 | 12 B | 18 B | `5 2` | 1.3 | 0.7 | 0.6 | 0.8 | 0.4 | 0.3 |
| example_01 | 25 B | 100 B | `10 5` | 1.3 | 0.7 | 0.5 | 0.7 | 0.4 | 0.3 |
| small_N_03 | 16 B | 10 B | `1 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_N_01 | 39 B | 35 B | `6 3` | 1.1 | 0.8 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_N_00 | 40 B | 50 B | `5 3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_N_04 | 16 B | 16 B | `4 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_N_02 | 40 B | 48 B | `8 3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
