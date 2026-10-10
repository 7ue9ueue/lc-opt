# compositional_inverse_of_formal_power_series

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 2.7.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 2.5.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| max_random_02 | 77.2 KB | 77.2 KB | `8000` | 2.7 | 0.7 | 2.0 | 2.4 | 0.5 | 1.9 |
| max_random_01 | 77.3 KB | 77.3 KB | `8000` | 2.7 | 0.8 | 2.0 | 2.4 | 0.5 | 1.9 |
| random_04 | 64.5 KB | 64.5 KB | `6677` | 2.7 | 0.7 | 2.0 | 2.3 | 0.5 | 1.9 |
| max_random_00 | 77.3 KB | 77.2 KB | `8000` | 2.7 | 0.7 | 2.0 | 2.3 | 0.5 | 1.8 |
| max_random_04 | 77.2 KB | 77.3 KB | `8000` | 2.7 | 0.7 | 2.0 | 2.3 | 0.5 | 1.9 |
| max_random_03 | 77.3 KB | 77.2 KB | `8000` | 2.7 | 0.8 | 1.9 | 2.4 | 0.5 | 1.9 |
| random_00 | 46.2 KB | 46.2 KB | `4790` | 2.7 | 0.8 | 1.9 | 2.3 | 0.5 | 1.8 |
| max_identity_00 | 15.6 KB | 15.6 KB | `8000` | 2.6 | 0.7 | 1.9 | 2.4 | 0.4 | 1.9 |
| random_01 | 41.5 KB | 41.5 KB | `4295` | 2.6 | 0.7 | 1.9 | 2.3 | 0.5 | 1.8 |
| random_02 | 40.4 KB | 40.4 KB | `4185` | 2.6 | 0.7 | 1.9 | 2.3 | 0.5 | 1.8 |
| random_03 | 31.5 KB | 31.5 KB | `3266` | 1.9 | 0.8 | 1.1 | 1.5 | 0.5 | 1.0 |
| example_00 | 12 B | 26 B | `5` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_08 | 92 B | 91 B | `10` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_06 | 73 B | 71 B | `8` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_05 | 64 B | 60 B | `7` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_09 | 105 B | 101 B | `11` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_03 | 44 B | 42 B | `5` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| identity_00 | 6 B | 4 B | `2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_02 | 32 B | 32 B | `4` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_07 | 84 B | 81 B | `9` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_00 | 14 B | 12 B | `2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_01 | 24 B | 22 B | `3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_04 | 54 B | 52 B | `6` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
