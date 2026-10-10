# min_plus_convolution_convex_arbitrary

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 10.4.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 10.1.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small_slopes_00 | `N M = 524288 524288` | 9.9 MB | 10.0 MB | 10.2 | 6.0 | 4.2 | 8.8 | 4.9 | 3.9 |
| only_first_small_01 | `N M = 524288 524288` | 10.0 MB | 10.5 MB | 10.0 | 6.3 | 3.7 | 9.4 | 5.0 | 4.4 |
| small_slopes_01 | `N M = 524288 524288` | 9.9 MB | 10.0 MB | 9.8 | 6.1 | 3.8 | 9.0 | 5.1 | 4.0 |
| only_first_small_00 | `N M = 524288 524288` | 10.0 MB | 10.5 MB | 9.7 | 6.4 | 3.4 | 9.5 | 5.3 | 4.2 |
| max_random_01 | `N M = 524288 524288` | 9.9 MB | 10.0 MB | 9.7 | 6.4 | 3.2 | 8.3 | 5.1 | 3.2 |
| monotone_03 | `N M = 524288 524288` | 9.7 MB | 10.1 MB | 9.6 | 6.4 | 3.2 | 8.6 | 5.0 | 3.6 |
| monotone_00 | `N M = 524288 524288` | 9.7 MB | 10.2 MB | 9.5 | 6.3 | 3.3 | 8.9 | 5.2 | 3.7 |
| max_random_00 | `N M = 524288 524288` | 9.9 MB | 10.0 MB | 9.4 | 6.1 | 3.3 | 8.3 | 5.0 | 3.4 |
| max_random_02 | `N M = 524288 524288` | 9.8 MB | 9.4 MB | 9.2 | 5.7 | 3.5 | 8.7 | 4.9 | 3.8 |
| monotone_02 | `N M = 524288 524288` | 9.7 MB | 9.7 MB | 9.0 | 6.0 | 3.0 | 8.0 | 4.8 | 3.2 |
| monotone_01 | `N M = 524288 524288` | 9.7 MB | 9.7 MB | 8.9 | 6.5 | 2.3 | 7.9 | 5.1 | 2.8 |
| random_01 | `N M = 463046 412907` | 8.3 MB | 8.4 MB | 8.2 | 5.3 | 2.9 | 7.1 | 4.3 | 2.8 |
| random_00 | `N M = 389813 410923` | 7.6 MB | 7.6 MB | 7.5 | 4.7 | 2.8 | 6.6 | 3.7 | 2.9 |
| near_power_of_2_00 | `N M = 262143 262143` | 5.0 MB | 5.0 MB | 5.3 | 3.5 | 1.8 | 4.4 | 2.8 | 1.6 |
| large_small_03 | `N M = 2 524288` | 4.9 MB | 5.2 MB | 5.3 | 3.4 | 1.9 | 4.5 | 2.7 | 1.8 |
| near_power_of_2_02 | `N M = 262143 262145` | 5.0 MB | 5.0 MB | 5.3 | 3.4 | 1.9 | 4.5 | 2.5 | 2.0 |
| near_power_of_2_01 | `N M = 262143 262144` | 4.9 MB | 4.7 MB | 5.3 | 3.3 | 2.0 | 4.6 | 2.6 | 2.1 |
| near_power_of_2_03 | `N M = 262144 262143` | 5.0 MB | 5.0 MB | 5.3 | 3.4 | 1.9 | 4.5 | 2.5 | 1.9 |
| near_power_of_2_05 | `N M = 262144 262145` | 5.0 MB | 5.0 MB | 5.3 | 3.4 | 1.9 | 4.5 | 2.6 | 1.9 |
| near_power_of_2_07 | `N M = 262145 262144` | 4.9 MB | 4.7 MB | 5.2 | 3.3 | 2.0 | 4.6 | 2.4 | 2.1 |
| near_power_of_2_04 | `N M = 262144 262144` | 4.9 MB | 4.7 MB | 5.2 | 3.3 | 2.0 | 4.5 | 2.6 | 2.0 |
| near_power_of_2_06 | `N M = 262145 262143` | 5.0 MB | 5.0 MB | 5.2 | 3.3 | 1.9 | 4.4 | 2.7 | 1.7 |
| near_power_of_2_08 | `N M = 262145 262145` | 4.9 MB | 4.7 MB | 5.2 | 3.2 | 2.0 | 4.6 | 2.4 | 2.1 |
| large_small_02 | `N M = 1 524288` | 4.9 MB | 5.3 MB | 5.2 | 3.5 | 1.7 | 4.2 | 2.7 | 1.6 |
| large_small_00 | `N M = 524288 1` | 5.0 MB | 5.5 MB | 4.9 | 3.5 | 1.4 | 4.1 | 2.8 | 1.3 |
| random_02 | `N M = 53336 382347` | 4.1 MB | 4.2 MB | 4.8 | 2.9 | 1.9 | 4.1 | 2.3 | 1.8 |
| large_small_01 | `N M = 524288 2` | 4.7 MB | 5.0 MB | 4.6 | 3.3 | 1.3 | 3.8 | 2.7 | 1.1 |
| example_00 | `N M = 4 5` | 22 B | 16 B | 1.3 | 0.8 | 0.6 | 0.9 | 0.5 | 0.4 |
| hack_00 | `N M = 17 8` | 104 B | 97 B | 1.2 | 0.7 | 0.4 | 0.8 | 0.5 | 0.3 |
| med_random_00 | `N M = 792 398` | 11.4 KB | 11.1 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_01 | `N M = 1 2` | 34 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_03 | `N M = 2 1` | 33 B | 21 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_05 | `N M = 2 3` | 54 B | 41 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| med_random_01 | `N M = 297 334` | 6.1 KB | 6.0 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| med_random_02 | `N M = 187 494` | 6.5 KB | 6.2 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_00 | `N M = 1 1` | 24 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_07 | `N M = 3 2` | 53 B | 41 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_04 | `N M = 2 2` | 44 B | 32 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_08 | `N M = 3 3` | 63 B | 51 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_02 | `N M = 1 3` | 42 B | 31 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_06 | `N M = 3 1` | 44 B | 33 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
