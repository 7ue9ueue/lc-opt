# sqrt_of_formal_power_series

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.1.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 11.8.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| max_random_01 | 4.7 MB | 4.7 MB | `500000` | 12.1 | 3.8 | 8.3 | 11.6 | 2.5 | 9.1 |
| random_01 | 4.7 MB | 4.7 MB | `500000` | 12.0 | 3.8 | 8.2 | 11.6 | 2.6 | 9.0 |
| max_random_02 | 4.7 MB | 4.7 MB | `500000` | 12.0 | 3.8 | 8.2 | 11.5 | 2.5 | 9.0 |
| random_02 | 4.7 MB | 4.7 MB | `500000` | 11.9 | 3.8 | 8.1 | 11.6 | 2.5 | 9.1 |
| monomial_02 | 976.6 KB | 976.6 KB | `500000` | 11.5 | 1.3 | 10.2 | 11.3 | 0.8 | 10.4 |
| lower_deg_zero_00 | 3.6 MB | 4.1 MB | `500000` | 10.5 | 3.4 | 7.1 | 9.8 | 2.3 | 7.5 |
| near_262144_02 | 2.5 MB | 2.5 MB | `262145` | 8.6 | 2.3 | 6.2 | 8.5 | 1.5 | 7.0 |
| monomial_00 | 976.6 KB | 976.6 KB | `500000` | 7.9 | 1.3 | 6.6 | 6.9 | 0.8 | 6.1 |
| lower_deg_zero_01 | 2.1 MB | 3.4 MB | `500000` | 7.4 | 2.8 | 4.7 | 6.2 | 1.8 | 4.4 |
| near_262144_01 | 2.5 MB | 2.5 MB | `262144` | 6.4 | 2.3 | 4.1 | 5.9 | 1.5 | 4.4 |
| all_zero_01 | 904.4 KB | 904.4 KB | `463046` | 4.5 | 1.3 | 3.2 | 2.9 | 0.8 | 2.1 |
| all_zero_00 | 761.4 KB | 761.4 KB | `389813` | 4.1 | 1.2 | 2.9 | 2.6 | 0.7 | 1.9 |
| lower_deg_zero_03 | 4.0 MB | 3 B | `500000` | 2.3 | 1.3 | 1.0 | 1.6 | 0.9 | 0.7 |
| lower_deg_zero_05 | 2.9 MB | 3 B | `500000` | 2.2 | 1.1 | 1.1 | 1.6 | 0.7 | 0.9 |
| max_random_00 | 4.7 MB | 3 B | `500000` | 2.2 | 1.4 | 0.8 | 1.5 | 0.9 | 0.6 |
| lower_deg_zero_06 | 2.5 MB | 3 B | `500000` | 2.2 | 1.1 | 1.1 | 1.6 | 0.7 | 0.9 |
| lower_deg_zero_02 | 2.7 MB | 3 B | `500000` | 2.1 | 1.1 | 1.0 | 1.5 | 0.7 | 0.8 |
| random_00 | 4.7 MB | 3 B | `500000` | 2.1 | 1.4 | 0.8 | 1.5 | 0.9 | 0.7 |
| lower_deg_zero_07 | 2.0 MB | 3 B | `500000` | 2.1 | 1.0 | 1.1 | 1.5 | 0.6 | 0.9 |
| lower_deg_zero_04 | 1.3 MB | 3 B | `500000` | 2.0 | 0.9 | 1.1 | 1.4 | 0.6 | 0.8 |
| monomial_01 | 976.6 KB | 3 B | `500000` | 1.9 | 0.8 | 1.1 | 1.4 | 0.5 | 0.9 |
| monomial_03 | 976.6 KB | 3 B | `500000` | 1.9 | 0.8 | 1.1 | 1.4 | 0.5 | 0.8 |
| near_262144_00 | 2.5 MB | 3 B | `262143` | 1.7 | 1.0 | 0.6 | 1.2 | 0.7 | 0.4 |
| example_00 | 11 B | 16 B | `4` | 1.2 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| example_01 | 12 B | 3 B | `4` | 1.1 | 0.8 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_03 | 42 B | 39 B | `4` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_00 | 12 B | 3 B | `1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_06 | 71 B | 3 B | `7` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_05 | 62 B | 58 B | `6` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_01 | 22 B | 20 B | `2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_02 | 30 B | 29 B | `3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_04 | 52 B | 49 B | `5` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_09 | 103 B | 3 B | `10` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_degree_07 | 82 B | 3 B | `8` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_08 | 89 B | 90 B | `9` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
