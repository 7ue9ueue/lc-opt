# min_plus_convolution_convex_convex

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 8.9.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 7.4.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small_slopes_01 | `N M = 524288 524288` | 10.0 MB | 11.0 MB | 8.5 | 6.6 | 1.9 | 7.4 | 5.4 | 2.0 |
| monotone_03 | `N M = 524288 524288` | 9.5 MB | 10.0 MB | 8.4 | 5.9 | 2.5 | 6.9 | 4.8 | 2.1 |
| monotone_00 | `N M = 524288 524288` | 9.6 MB | 10.0 MB | 8.3 | 5.9 | 2.4 | 7.1 | 4.8 | 2.4 |
| max_random_01 | `N M = 524288 524288` | 10.0 MB | 10.3 MB | 8.3 | 6.4 | 1.9 | 7.2 | 5.3 | 1.9 |
| max_random_00 | `N M = 524288 524288` | 10.0 MB | 10.4 MB | 8.2 | 6.5 | 1.7 | 7.1 | 5.4 | 1.8 |
| small_slopes_00 | `N M = 524288 524288` | 10.0 MB | 10.0 MB | 8.1 | 6.0 | 2.0 | 6.8 | 4.8 | 2.1 |
| monotone_01 | `N M = 524288 524288` | 9.6 MB | 9.6 MB | 8.1 | 5.8 | 2.3 | 7.2 | 4.8 | 2.4 |
| monotone_02 | `N M = 524288 524288` | 9.5 MB | 9.6 MB | 8.1 | 5.9 | 2.2 | 6.9 | 4.7 | 2.2 |
| max_random_02 | `N M = 524288 524288` | 9.7 MB | 10.0 MB | 7.9 | 6.1 | 1.7 | 6.9 | 4.9 | 2.0 |
| random_01 | `N M = 463046 412907` | 8.4 MB | 8.6 MB | 7.1 | 5.7 | 1.4 | 6.1 | 4.3 | 1.8 |
| random_00 | `N M = 389813 410923` | 7.6 MB | 7.9 MB | 6.4 | 5.0 | 1.4 | 5.6 | 4.0 | 1.6 |
| near_power_of_2_03 | `N M = 262144 262143` | 5.0 MB | 5.2 MB | 4.7 | 3.4 | 1.3 | 4.2 | 2.7 | 1.4 |
| near_power_of_2_05 | `N M = 262144 262145` | 5.0 MB | 5.1 MB | 4.7 | 3.9 | 0.8 | 3.9 | 2.7 | 1.2 |
| near_power_of_2_00 | `N M = 262143 262143` | 4.9 MB | 5.1 MB | 4.6 | 3.3 | 1.3 | 4.1 | 2.8 | 1.3 |
| near_power_of_2_07 | `N M = 262145 262144` | 4.9 MB | 5.1 MB | 4.6 | 3.6 | 1.0 | 3.8 | 2.6 | 1.2 |
| near_power_of_2_02 | `N M = 262143 262145` | 4.9 MB | 5.1 MB | 4.6 | 3.4 | 1.2 | 4.0 | 2.7 | 1.2 |
| near_power_of_2_01 | `N M = 262143 262144` | 4.9 MB | 5.1 MB | 4.6 | 3.4 | 1.2 | 3.9 | 2.6 | 1.3 |
| near_power_of_2_06 | `N M = 262145 262143` | 5.0 MB | 5.2 MB | 4.6 | 3.6 | 1.0 | 4.0 | 2.7 | 1.3 |
| near_power_of_2_08 | `N M = 262145 262145` | 4.9 MB | 5.2 MB | 4.6 | 3.5 | 1.1 | 4.0 | 2.7 | 1.3 |
| near_power_of_2_04 | `N M = 262144 262144` | 4.8 MB | 5.0 MB | 4.6 | 3.3 | 1.3 | 4.0 | 2.7 | 1.3 |
| random_02 | `N M = 53336 382347` | 4.2 MB | 4.3 MB | 4.0 | 3.2 | 0.9 | 3.3 | 2.3 | 1.0 |
| example_00 | `N M = 4 5` | 22 B | 16 B | 1.4 | 0.8 | 0.6 | 0.8 | 0.5 | 0.3 |
| med_random_00 | `N M = 792 398` | 11.4 KB | 11.9 KB | 1.2 | 0.8 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_03 | `N M = 2 1` | 33 B | 21 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| med_random_01 | `N M = 297 334` | 6.1 KB | 6.3 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_00 | `N M = 1 1` | 24 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| med_random_02 | `N M = 187 494` | 6.6 KB | 6.8 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_06 | `N M = 3 1` | 44 B | 33 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_01 | `N M = 1 2` | 34 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_05 | `N M = 2 3` | 54 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_02 | `N M = 1 3` | 42 B | 32 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_08 | `N M = 3 3` | 63 B | 53 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_07 | `N M = 3 2` | 53 B | 41 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_04 | `N M = 2 2` | 44 B | 32 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
