# exp_of_formal_power_series

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 16.8.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 17.6.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| max_ans_zero_00 | 4.7 MB | 4.7 MB | `500000` | 16.8 | 3.9 | 12.9 | 17.4 | 2.6 | 14.7 |
| max_random_00 | 4.7 MB | 4.7 MB | `500000` | 16.7 | 3.8 | 12.9 | 17.3 | 2.6 | 14.6 |
| max_random_01 | 4.7 MB | 4.7 MB | `500000` | 16.7 | 3.8 | 12.9 | 17.3 | 2.6 | 14.7 |
| max_random_03 | 4.7 MB | 4.7 MB | `500000` | 16.7 | 3.9 | 12.8 | 17.3 | 2.7 | 14.6 |
| max_random_02 | 4.7 MB | 4.7 MB | `500000` | 16.6 | 3.8 | 12.9 | 17.3 | 2.7 | 14.6 |
| max_random_04 | 4.7 MB | 4.7 MB | `500000` | 16.6 | 3.7 | 12.9 | 17.5 | 2.6 | 14.8 |
| max_all_zero_00 | 976.6 KB | 976.6 KB | `500000` | 16.6 | 1.4 | 15.2 | 17.4 | 0.9 | 16.5 |
| random_01 | 4.4 MB | 4.4 MB | `463046` | 16.5 | 3.5 | 13.0 | 17.1 | 2.5 | 14.6 |
| random_03 | 4.0 MB | 4.0 MB | `429249` | 16.2 | 3.0 | 13.2 | 17.0 | 2.3 | 14.6 |
| random_00 | 3.7 MB | 3.7 MB | `389813` | 12.9 | 3.1 | 9.8 | 13.1 | 2.1 | 11.0 |
| random_04 | 2.6 MB | 2.6 MB | `277012` | 12.0 | 2.2 | 9.8 | 12.3 | 1.7 | 10.6 |
| near_262144_02 | 2.5 MB | 2.5 MB | `262145` | 12.0 | 2.3 | 9.6 | 12.5 | 1.6 | 10.9 |
| near_262144_00 | 2.5 MB | 2.5 MB | `262143` | 8.9 | 2.3 | 6.6 | 9.0 | 1.6 | 7.4 |
| near_262144_01 | 2.5 MB | 2.5 MB | `262144` | 8.8 | 2.3 | 6.5 | 9.1 | 1.6 | 7.5 |
| random_02 | 515.1 KB | 514.9 KB | `53336` | 2.8 | 1.0 | 1.8 | 2.4 | 0.7 | 1.8 |
| example_00 | 12 B | 34 B | `5` | 1.2 | 0.7 | 0.5 | 0.7 | 0.5 | 0.3 |
| small_degree_00 | 4 B | 2 B | `1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_05 | 54 B | 51 B | `6` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_03 | 34 B | 32 B | `4` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_01 | 14 B | 12 B | `2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_04 | 44 B | 41 B | `5` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_09 | 95 B | 91 B | `10` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_08 | 81 B | 79 B | `9` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_02 | 24 B | 22 B | `3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_07 | 74 B | 72 B | `8` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_06 | 63 B | 61 B | `7` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
