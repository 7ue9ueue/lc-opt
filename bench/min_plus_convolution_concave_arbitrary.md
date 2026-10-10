# min_plus_convolution_concave_arbitrary

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 27.6.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 22.8.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| monotone_02 | `N M = 524288 524288` | 9.9 MB | 10.4 MB | 27.6 | 7.9 | 19.7 | 22.7 | 5.4 | 17.2 |
| monotone_01 | `N M = 524288 524288` | 9.9 MB | 10.4 MB | 26.9 | 7.9 | 19.0 | 22.4 | 5.5 | 16.9 |
| monotone_03 | `N M = 524288 524288` | 9.9 MB | 9.9 MB | 12.0 | 6.9 | 5.0 | 12.0 | 5.0 | 7.1 |
| small_slopes_01 | `N M = 524288 524288` | 9.4 MB | 9.0 MB | 11.9 | 5.6 | 6.3 | 10.2 | 4.6 | 5.5 |
| monotone_00 | `N M = 524288 524288` | 9.9 MB | 9.9 MB | 11.8 | 8.5 | 3.3 | 10.9 | 5.1 | 5.8 |
| small_slopes_00 | `N M = 524288 524288` | 9.9 MB | 10.0 MB | 11.2 | 6.0 | 5.2 | 10.0 | 4.9 | 5.1 |
| max_random_01 | `N M = 524288 524288` | 9.9 MB | 9.0 MB | 11.1 | 6.2 | 5.0 | 10.1 | 4.7 | 5.4 |
| only_first_small_00 | `N M = 524288 524288` | 10.0 MB | 10.0 MB | 10.9 | 6.3 | 4.6 | 10.1 | 5.2 | 4.9 |
| max_random_02 | `N M = 524288 524288` | 9.9 MB | 10.0 MB | 10.9 | 6.6 | 4.3 | 10.1 | 5.1 | 5.0 |
| only_first_small_01 | `N M = 524288 524288` | 10.0 MB | 10.0 MB | 10.8 | 6.0 | 4.8 | 10.5 | 4.9 | 5.6 |
| max_random_00 | `N M = 524288 524288` | 9.9 MB | 9.0 MB | 10.8 | 6.1 | 4.7 | 10.1 | 4.7 | 5.3 |
| large_small_03 | `N M = 2 524288` | 4.9 MB | 5.1 MB | 10.6 | 3.6 | 6.9 | 10.3 | 2.7 | 7.5 |
| random_00 | `N M = 389813 410923` | 7.6 MB | 7.6 MB | 10.5 | 4.8 | 5.7 | 8.0 | 3.9 | 4.0 |
| random_01 | `N M = 463046 412907` | 8.3 MB | 7.5 MB | 9.4 | 4.9 | 4.5 | 7.8 | 3.9 | 3.9 |
| near_power_of_2_05 | `N M = 262144 262145` | 5.0 MB | 5.0 MB | 8.8 | 3.8 | 4.9 | 6.2 | 2.8 | 3.4 |
| large_small_02 | `N M = 1 524288` | 4.9 MB | 5.2 MB | 8.3 | 3.6 | 4.7 | 9.5 | 2.7 | 6.7 |
| large_small_01 | `N M = 524288 2` | 5.0 MB | 5.5 MB | 7.5 | 3.6 | 3.9 | 5.4 | 2.7 | 2.7 |
| near_power_of_2_02 | `N M = 262143 262145` | 5.0 MB | 4.5 MB | 7.4 | 3.5 | 3.9 | 6.6 | 2.6 | 4.0 |
| near_power_of_2_01 | `N M = 262143 262144` | 5.0 MB | 5.0 MB | 7.4 | 3.7 | 3.7 | 6.6 | 2.7 | 3.8 |
| large_small_00 | `N M = 524288 1` | 5.0 MB | 5.2 MB | 7.2 | 3.4 | 3.8 | 5.2 | 2.7 | 2.4 |
| near_power_of_2_04 | `N M = 262144 262144` | 5.0 MB | 5.0 MB | 7.2 | 4.0 | 3.2 | 5.5 | 2.7 | 2.8 |
| random_02 | `N M = 53336 382347` | 4.1 MB | 3.7 MB | 6.5 | 2.8 | 3.7 | 5.1 | 2.0 | 3.0 |
| near_power_of_2_06 | `N M = 262145 262143` | 5.0 MB | 4.5 MB | 6.3 | 3.4 | 2.9 | 4.8 | 2.6 | 2.3 |
| near_power_of_2_03 | `N M = 262144 262143` | 5.0 MB | 4.5 MB | 6.3 | 3.7 | 2.6 | 5.6 | 2.7 | 2.9 |
| near_power_of_2_07 | `N M = 262145 262144` | 5.0 MB | 5.0 MB | 6.2 | 3.5 | 2.7 | 4.9 | 2.7 | 2.2 |
| near_power_of_2_08 | `N M = 262145 262145` | 5.0 MB | 5.0 MB | 6.2 | 3.3 | 2.8 | 5.1 | 2.7 | 2.3 |
| near_power_of_2_00 | `N M = 262143 262143` | 5.0 MB | 5.0 MB | 6.1 | 3.9 | 2.2 | 4.9 | 2.8 | 2.2 |
| example_00 | `N M = 4 5` | 22 B | 16 B | 1.6 | 0.8 | 0.8 | 1.2 | 0.5 | 0.8 |
| med_random_00 | `N M = 792 398` | 11.6 KB | 11.6 KB | 1.4 | 0.7 | 0.7 | 1.1 | 0.5 | 0.6 |
| hack_00 | `N M = 17 8` | 114 B | 108 B | 1.4 | 0.7 | 0.6 | 1.1 | 0.5 | 0.7 |
| small_00 | `N M = 1 1` | 24 B | 10 B | 1.4 | 0.7 | 0.7 | 1.1 | 0.5 | 0.7 |
| small_04 | `N M = 2 2` | 43 B | 31 B | 1.4 | 0.7 | 0.7 | 1.1 | 0.5 | 0.6 |
| small_01 | `N M = 1 2` | 34 B | 20 B | 1.4 | 0.7 | 0.6 | 1.1 | 0.5 | 0.6 |
| med_random_02 | `N M = 187 494` | 6.6 KB | 6.6 KB | 1.4 | 0.7 | 0.6 | 1.1 | 0.5 | 0.6 |
| small_06 | `N M = 3 1` | 44 B | 30 B | 1.4 | 0.8 | 0.6 | 1.1 | 0.5 | 0.6 |
| small_05 | `N M = 2 3` | 54 B | 44 B | 1.4 | 0.8 | 0.6 | 1.0 | 0.5 | 0.6 |
| small_03 | `N M = 2 1` | 34 B | 22 B | 1.4 | 0.7 | 0.6 | 1.1 | 0.5 | 0.6 |
| med_random_01 | `N M = 297 334` | 6.1 KB | 6.1 KB | 1.4 | 0.7 | 0.6 | 1.1 | 0.5 | 0.6 |
| small_02 | `N M = 1 3` | 42 B | 30 B | 1.3 | 0.7 | 0.6 | 1.1 | 0.5 | 0.6 |
| small_08 | `N M = 3 3` | 63 B | 52 B | 1.3 | 0.7 | 0.6 | 1.1 | 0.5 | 0.6 |
| small_07 | `N M = 3 2` | 51 B | 39 B | 1.3 | 0.7 | 0.6 | 1.1 | 0.5 | 0.6 |
