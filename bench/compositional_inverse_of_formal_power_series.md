# compositional_inverse_of_formal_power_series

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 2.7. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 2.5. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_random_02 | `N = 8000` | 77.2 KB | 77.2 KB | 2.7 | 0.7 | 2.0 | 2.4 | 0.5 | 1.9 |
| max_random_01 | `N = 8000` | 77.3 KB | 77.3 KB | 2.7 | 0.8 | 2.0 | 2.4 | 0.5 | 1.9 |
| random_04 | `N = 6677` | 64.5 KB | 64.5 KB | 2.7 | 0.7 | 2.0 | 2.3 | 0.5 | 1.9 |
| max_random_00 | `N = 8000` | 77.3 KB | 77.2 KB | 2.7 | 0.7 | 2.0 | 2.3 | 0.5 | 1.8 |
| max_random_04 | `N = 8000` | 77.2 KB | 77.3 KB | 2.7 | 0.7 | 2.0 | 2.3 | 0.5 | 1.9 |
| max_random_03 | `N = 8000` | 77.3 KB | 77.2 KB | 2.7 | 0.8 | 1.9 | 2.4 | 0.5 | 1.9 |
| random_00 | `N = 4790` | 46.2 KB | 46.2 KB | 2.7 | 0.8 | 1.9 | 2.3 | 0.5 | 1.8 |
| max_identity_00 | `N = 8000` | 15.6 KB | 15.6 KB | 2.6 | 0.7 | 1.9 | 2.4 | 0.4 | 1.9 |
| random_01 | `N = 4295` | 41.5 KB | 41.5 KB | 2.6 | 0.7 | 1.9 | 2.3 | 0.5 | 1.8 |
| random_02 | `N = 4185` | 40.4 KB | 40.4 KB | 2.6 | 0.7 | 1.9 | 2.3 | 0.5 | 1.8 |
| random_03 | `N = 3266` | 31.5 KB | 31.5 KB | 1.9 | 0.8 | 1.1 | 1.5 | 0.5 | 1.0 |
| example_00 | `N = 5` | 12 B | 26 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_08 | `N = 10` | 92 B | 91 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_06 | `N = 8` | 73 B | 71 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_05 | `N = 7` | 64 B | 60 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_09 | `N = 11` | 105 B | 101 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_03 | `N = 5` | 44 B | 42 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| identity_00 | `N = 2` | 6 B | 4 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_degree_02 | `N = 4` | 32 B | 32 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_07 | `N = 9` | 84 B | 81 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_degree_00 | `N = 2` | 14 B | 12 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_01 | `N = 3` | 24 B | 22 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_degree_04 | `N = 6` | 54 B | 52 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
