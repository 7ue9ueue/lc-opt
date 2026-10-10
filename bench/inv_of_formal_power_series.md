# inv_of_formal_power_series

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.0.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.0.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| max_random_00 | 4.7 MB | 4.7 MB | `500000` | 12.0 | 3.8 | 8.2 | 12.0 | 2.5 | 9.5 |
| max_random_03 | 4.7 MB | 4.7 MB | `500000` | 12.0 | 3.8 | 8.1 | 11.8 | 2.6 | 9.2 |
| max_random_01 | 4.7 MB | 4.7 MB | `500000` | 12.0 | 3.9 | 8.0 | 11.6 | 2.6 | 9.0 |
| max_random_02 | 4.7 MB | 4.7 MB | `500000` | 11.9 | 3.8 | 8.1 | 11.6 | 2.5 | 9.0 |
| max_random_or_zero_00 | 4.7 MB | 4.7 MB | `500000` | 11.9 | 3.8 | 8.1 | 11.9 | 2.5 | 9.4 |
| max_random_04 | 4.7 MB | 4.7 MB | `500000` | 11.9 | 3.8 | 8.1 | 11.8 | 2.5 | 9.3 |
| random_01 | 4.4 MB | 4.4 MB | `463046` | 11.7 | 3.6 | 8.2 | 11.6 | 2.3 | 9.3 |
| random_03 | 4.0 MB | 4.0 MB | `429249` | 11.4 | 3.3 | 8.1 | 11.5 | 2.2 | 9.3 |
| random_00 | 3.7 MB | 3.7 MB | `389813` | 11.2 | 3.1 | 8.1 | 11.3 | 2.1 | 9.3 |
| random_04 | 2.6 MB | 2.6 MB | `277012` | 10.3 | 2.3 | 8.0 | 10.7 | 1.6 | 9.1 |
| near_262144_02 | 2.5 MB | 2.5 MB | `262145` | 10.3 | 2.3 | 7.9 | 10.7 | 1.5 | 9.1 |
| near_262144_00 | 2.5 MB | 2.5 MB | `262143` | 6.5 | 2.3 | 4.1 | 6.0 | 1.5 | 4.5 |
| near_262144_01 | 2.5 MB | 2.5 MB | `262144` | 6.4 | 2.3 | 4.2 | 6.0 | 1.5 | 4.5 |
| random_02 | 515.1 KB | 515.1 KB | `53336` | 2.3 | 1.0 | 1.2 | 1.9 | 0.7 | 1.2 |
| example_00 | 12 B | 50 B | `5` | 1.3 | 0.7 | 0.6 | 0.7 | 0.5 | 0.2 |
| small_degree_09 | 103 B | 100 B | `10` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_06 | 71 B | 69 B | `7` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_00 | 12 B | 10 B | `1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_03 | 42 B | 40 B | `4` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_degree_05 | 62 B | 59 B | `6` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_01 | 22 B | 20 B | `2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_02 | 30 B | 30 B | `3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_degree_08 | 89 B | 88 B | `9` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_04 | 52 B | 49 B | `5` | 1.1 | 0.7 | 0.3 | 0.7 | 0.5 | 0.2 |
| small_degree_07 | 82 B | 80 B | `8` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
