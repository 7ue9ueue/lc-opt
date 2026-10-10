# log_of_formal_power_series

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 14.4.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 14.3.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| max_random_00 | 4.7 MB | 4.7 MB | `500000` | 14.3 | 3.8 | 10.5 | 14.1 | 2.5 | 11.6 |
| max_random_01 | 4.7 MB | 4.7 MB | `500000` | 14.3 | 3.8 | 10.5 | 14.0 | 2.5 | 11.5 |
| random_01 | 4.4 MB | 4.4 MB | `463046` | 14.1 | 3.5 | 10.6 | 14.0 | 2.4 | 11.6 |
| max_random_02 | 4.7 MB | 4.7 MB | `500000` | 14.1 | 3.8 | 10.4 | 14.2 | 2.6 | 11.6 |
| max_random_03 | 4.7 MB | 4.7 MB | `500000` | 14.1 | 3.8 | 10.3 | 14.3 | 2.7 | 11.6 |
| max_random_04 | 4.7 MB | 4.7 MB | `500000` | 14.1 | 3.7 | 10.3 | 14.3 | 2.6 | 11.6 |
| max_all_zero_00 | 976.6 KB | 976.6 KB | `500000` | 13.9 | 1.3 | 12.6 | 14.2 | 0.8 | 13.4 |
| random_03 | 4.0 MB | 4.0 MB | `429249` | 13.7 | 3.4 | 10.3 | 13.9 | 2.3 | 11.6 |
| random_00 | 3.7 MB | 3.7 MB | `389813` | 11.1 | 3.1 | 8.0 | 10.8 | 2.1 | 8.7 |
| random_04 | 2.6 MB | 2.6 MB | `277012` | 10.0 | 2.4 | 7.6 | 10.1 | 1.6 | 8.6 |
| near_262144_02 | 2.5 MB | 2.5 MB | `262145` | 7.6 | 2.3 | 5.3 | 7.3 | 1.5 | 5.8 |
| near_262144_00 | 2.5 MB | 2.5 MB | `262143` | 7.5 | 2.3 | 5.2 | 7.3 | 1.6 | 5.7 |
| near_262144_01 | 2.5 MB | 2.5 MB | `262144` | 7.5 | 2.3 | 5.1 | 7.3 | 1.6 | 5.8 |
| random_02 | 515.1 KB | 515.0 KB | `53336` | 2.6 | 1.0 | 1.5 | 2.1 | 0.7 | 1.5 |
| example_00 | 36 B | 10 B | `5` | 1.3 | 0.7 | 0.6 | 0.7 | 0.5 | 0.3 |
| small_degree_08 | 81 B | 82 B | `9` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_00 | 4 B | 2 B | `1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_06 | 63 B | 62 B | `7` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_01 | 14 B | 12 B | `2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_05 | 54 B | 50 B | `6` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_02 | 24 B | 22 B | `3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_03 | 34 B | 32 B | `4` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_04 | 44 B | 42 B | `5` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_09 | 95 B | 90 B | `10` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_07 | 74 B | 71 B | `8` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
