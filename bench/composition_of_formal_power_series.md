# composition_of_formal_power_series

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 2.9. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 2.6. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hack2_00 | `N = 8000` | 154.5 KB | 77.2 KB | 2.9 | 0.8 | 2.1 | 2.6 | 0.5 | 2.1 |
| hack2_01 | `N = 8000` | 154.5 KB | 77.2 KB | 2.8 | 0.8 | 2.1 | 2.5 | 0.5 | 2.0 |
| hack2_02 | `N = 8000` | 154.4 KB | 77.2 KB | 2.8 | 0.8 | 2.1 | 2.5 | 0.5 | 2.0 |
| max_random_01 | `N = 8000` | 154.6 KB | 77.2 KB | 2.8 | 0.8 | 2.1 | 2.5 | 0.5 | 2.1 |
| hack_00 | `N = 8000` | 151.9 KB | 74.7 KB | 2.8 | 0.8 | 2.1 | 2.6 | 0.5 | 2.1 |
| max_random_04 | `N = 8000` | 154.4 KB | 77.2 KB | 2.8 | 0.8 | 2.0 | 2.5 | 0.5 | 2.0 |
| max_random_03 | `N = 8000` | 154.5 KB | 77.3 KB | 2.8 | 0.8 | 2.1 | 2.6 | 0.5 | 2.1 |
| max_random_02 | `N = 8000` | 154.5 KB | 77.3 KB | 2.8 | 0.8 | 2.0 | 2.5 | 0.5 | 2.0 |
| max_random_00 | `N = 8000` | 154.5 KB | 77.3 KB | 2.8 | 0.8 | 2.0 | 2.5 | 0.5 | 2.0 |
| hack_02 | `N = 8000` | 124.6 KB | 47.4 KB | 2.8 | 0.8 | 2.1 | 2.5 | 0.5 | 2.1 |
| random_04 | `N = 6676` | 128.9 KB | 64.5 KB | 2.8 | 0.8 | 2.0 | 2.5 | 0.5 | 2.0 |
| hack_01 | `N = 8000` | 118.8 KB | 41.5 KB | 2.8 | 0.8 | 2.0 | 2.5 | 0.5 | 2.1 |
| random_00 | `N = 4789` | 92.5 KB | 46.2 KB | 2.7 | 0.8 | 1.9 | 2.4 | 0.5 | 2.0 |
| random_02 | `N = 4184` | 80.8 KB | 40.4 KB | 2.7 | 0.7 | 2.0 | 2.4 | 0.5 | 2.0 |
| random_01 | `N = 4294` | 83.0 KB | 41.5 KB | 2.7 | 0.7 | 2.0 | 2.4 | 0.5 | 1.9 |
| random_03 | `N = 3265` | 63.1 KB | 31.5 KB | 1.9 | 0.7 | 1.1 | 1.5 | 0.5 | 1.0 |
| example_00 | `N = 5` | 22 B | 13 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_08 | `N = 7` | 131 B | 68 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_06 | `N = 6` | 112 B | 59 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_09 | `N = 1` | 14 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_00 | `N = 5` | 94 B | 50 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_02 | `N = 8` | 152 B | 79 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_03 | `N = 1` | 14 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_04 | `N = 4` | 74 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_07 | `N = 3` | 54 B | 29 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_01 | `N = 6` | 113 B | 60 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_05 | `N = 10` | 191 B | 100 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
