# pow_of_formal_power_series

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 28.1. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 28.4. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_random_00 | `N M = 500000 715051129791443637` | 4.7 MB | 4.7 MB | 27.2 | 3.4 | 23.8 | 27.8 | 2.6 | 25.2 |
| max_random_01 | `N M = 500000 284482609428762822` | 4.7 MB | 4.7 MB | 26.6 | 3.6 | 23.0 | 27.9 | 2.6 | 25.3 |
| lower_deg_zero2_00 | `N M = 500000 100` | 4.7 MB | 4.2 MB | 26.5 | 3.1 | 23.4 | 27.6 | 2.4 | 25.2 |
| binary_exp_max_00 | `N M = 500000 576460752303423487` | 4.7 MB | 4.7 MB | 26.4 | 3.5 | 22.8 | 27.9 | 2.7 | 25.2 |
| max_random_02 | `N M = 500000 731950446832832600` | 4.7 MB | 4.7 MB | 26.3 | 3.5 | 22.8 | 27.7 | 2.7 | 25.0 |
| random_01 | `N M = 463046 376989097743764715` | 4.4 MB | 4.4 MB | 25.9 | 3.2 | 22.7 | 27.2 | 2.5 | 24.8 |
| lower_deg_zero2_01 | `N M = 500000 100` | 4.7 MB | 3.4 MB | 20.8 | 2.7 | 18.1 | 21.4 | 2.2 | 19.2 |
| lower_deg_zero2_02 | `N M = 500000 100` | 4.7 MB | 3.0 MB | 20.6 | 2.5 | 18.1 | 21.4 | 2.0 | 19.3 |
| monomial_ans_low_deg_03 | `N M = 429252 25838` | 838.4 KB | 838.4 KB | 19.8 | 1.2 | 18.6 | 20.8 | 0.8 | 20.0 |
| random_00 | `N M = 389813 747461874239661793` | 3.7 MB | 3.7 MB | 19.7 | 2.7 | 17.0 | 20.7 | 2.1 | 18.6 |
| monomial_ans_low_deg_00 | `N M = 389813 747461874239661792` | 761.4 KB | 761.4 KB | 19.7 | 1.2 | 18.5 | 20.4 | 0.7 | 19.7 |
| monomial_ans_low_deg_01 | `N M = 463047 412906` | 904.4 KB | 904.4 KB | 6.6 | 1.2 | 5.3 | 5.5 | 0.8 | 4.6 |
| lower_deg_zero2_03 | `N M = 500000 100` | 4.7 MB | 1.1 MB | 5.6 | 2.0 | 3.6 | 4.0 | 1.3 | 2.6 |
| lower_deg_zero_01 | `N M = 500000 284482609428762822` | 4.6 MB | 976.6 KB | 5.0 | 1.8 | 3.2 | 3.3 | 1.3 | 2.0 |
| lower_deg_zero_04 | `N M = 500000 247794345342417428` | 3.3 MB | 976.6 KB | 5.0 | 1.6 | 3.3 | 3.4 | 1.1 | 2.2 |
| monomial_00 | `N M = 500000 1000000000000000000` | 976.6 KB | 976.6 KB | 5.0 | 1.2 | 3.7 | 3.1 | 0.9 | 2.3 |
| lower_deg_zero_06 | `N M = 500000 314172295141061510` | 4.1 MB | 976.6 KB | 4.9 | 1.7 | 3.2 | 3.4 | 1.3 | 2.2 |
| lower_deg_zero_05 | `N M = 500000 708562095613202282` | 3.6 MB | 976.6 KB | 4.9 | 1.6 | 3.3 | 3.4 | 1.2 | 2.2 |
| lower_deg_zero_03 | `N M = 500000 57891326865607873` | 1.7 MB | 976.6 KB | 4.9 | 1.4 | 3.5 | 3.2 | 0.9 | 2.3 |
| lower_deg_zero_00 | `N M = 500000 715051129791443637` | 2.1 MB | 976.6 KB | 4.9 | 1.4 | 3.5 | 3.4 | 1.0 | 2.4 |
| lower_deg_zero_07 | `N M = 500000 241218520153159259` | 2.0 MB | 976.6 KB | 4.8 | 1.4 | 3.4 | 3.4 | 1.0 | 2.4 |
| lower_deg_zero_02 | `N M = 500000 731950446832832600` | 1.2 MB | 976.6 KB | 4.8 | 1.3 | 3.5 | 3.2 | 0.9 | 2.3 |
| monomial_01 | `N M = 500000 1000000000000000000` | 976.6 KB | 976.6 KB | 4.8 | 1.3 | 3.5 | 3.1 | 0.8 | 2.3 |
| M_zero_01 | `N M = 463046 0` | 4.4 MB | 904.4 KB | 4.8 | 1.8 | 2.9 | 3.2 | 1.2 | 2.0 |
| monomial_03 | `N M = 500000 1000000000000000000` | 976.6 KB | 976.6 KB | 4.7 | 1.3 | 3.4 | 3.1 | 0.8 | 2.3 |
| monomial_02 | `N M = 500000 1000000000000000000` | 976.6 KB | 976.6 KB | 4.7 | 1.2 | 3.4 | 3.1 | 0.8 | 2.3 |
| all_zero_01 | `N M = 463046 376989097743764715` | 904.4 KB | 904.4 KB | 4.5 | 1.3 | 3.2 | 3.0 | 0.8 | 2.2 |
| M_zero_00 | `N M = 389813 0` | 3.7 MB | 761.4 KB | 4.3 | 1.6 | 2.7 | 2.7 | 1.0 | 1.7 |
| all_zero_00 | `N M = 389813 747461874239661793` | 761.4 KB | 761.4 KB | 4.0 | 1.2 | 2.8 | 2.6 | 0.8 | 1.8 |
| random_02 | `N M = 53336 701295191615460747` | 515.1 KB | 515.1 KB | 3.9 | 1.0 | 2.9 | 3.5 | 0.6 | 2.8 |
| monomial_ans_low_deg_02 | `N M = 53338 21898` | 104.2 KB | 104.2 KB | 1.9 | 0.8 | 1.2 | 1.4 | 0.5 | 0.9 |
| example_00 | `N M = 4 3` | 13 B | 8 B | 1.3 | 0.7 | 0.6 | 0.7 | 0.5 | 0.3 |
| overflow_killer_00 | `N M = 10 536870912` | 33 B | 20 B | 1.3 | 0.7 | 0.5 | 0.7 | 0.5 | 0.2 |
| example_01 | `N M = 2 2` | 8 B | 4 B | 1.2 | 0.8 | 0.4 | 0.7 | 0.5 | 0.3 |
| example_02 | `N M = 2 0` | 8 B | 4 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| hack_00 | `N M = 1 2` | 6 B | 2 B | 1.1 | 0.8 | 0.4 | 0.7 | 0.5 | 0.2 |
| overflow_killer_01 | `N M = 33 576460752303423488` | 88 B | 66 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
