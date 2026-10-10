# min_plus_convolution_convex_convex

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 8.9.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 7.4.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| small_slopes_01 | 10.0 MB | 11.0 MB | `524288 524288` | 8.5 | 6.6 | 1.9 | 7.4 | 5.4 | 2.0 |
| monotone_03 | 9.5 MB | 10.0 MB | `524288 524288` | 8.4 | 5.9 | 2.5 | 6.9 | 4.8 | 2.1 |
| monotone_00 | 9.6 MB | 10.0 MB | `524288 524288` | 8.3 | 5.9 | 2.4 | 7.1 | 4.8 | 2.4 |
| max_random_01 | 10.0 MB | 10.3 MB | `524288 524288` | 8.3 | 6.4 | 1.9 | 7.2 | 5.3 | 1.9 |
| max_random_00 | 10.0 MB | 10.4 MB | `524288 524288` | 8.2 | 6.5 | 1.7 | 7.1 | 5.4 | 1.8 |
| small_slopes_00 | 10.0 MB | 10.0 MB | `524288 524288` | 8.1 | 6.0 | 2.0 | 6.8 | 4.8 | 2.1 |
| monotone_01 | 9.6 MB | 9.6 MB | `524288 524288` | 8.1 | 5.8 | 2.3 | 7.2 | 4.8 | 2.4 |
| monotone_02 | 9.5 MB | 9.6 MB | `524288 524288` | 8.1 | 5.9 | 2.2 | 6.9 | 4.7 | 2.2 |
| max_random_02 | 9.7 MB | 10.0 MB | `524288 524288` | 7.9 | 6.1 | 1.7 | 6.9 | 4.9 | 2.0 |
| random_01 | 8.4 MB | 8.6 MB | `463046 412907` | 7.1 | 5.7 | 1.4 | 6.1 | 4.3 | 1.8 |
| random_00 | 7.6 MB | 7.9 MB | `389813 410923` | 6.4 | 5.0 | 1.4 | 5.6 | 4.0 | 1.6 |
| near_power_of_2_03 | 5.0 MB | 5.2 MB | `262144 262143` | 4.7 | 3.4 | 1.3 | 4.2 | 2.7 | 1.4 |
| near_power_of_2_05 | 5.0 MB | 5.1 MB | `262144 262145` | 4.7 | 3.9 | 0.8 | 3.9 | 2.7 | 1.2 |
| near_power_of_2_00 | 4.9 MB | 5.1 MB | `262143 262143` | 4.6 | 3.3 | 1.3 | 4.1 | 2.8 | 1.3 |
| near_power_of_2_07 | 4.9 MB | 5.1 MB | `262145 262144` | 4.6 | 3.6 | 1.0 | 3.8 | 2.6 | 1.2 |
| near_power_of_2_02 | 4.9 MB | 5.1 MB | `262143 262145` | 4.6 | 3.4 | 1.2 | 4.0 | 2.7 | 1.2 |
| near_power_of_2_01 | 4.9 MB | 5.1 MB | `262143 262144` | 4.6 | 3.4 | 1.2 | 3.9 | 2.6 | 1.3 |
| near_power_of_2_06 | 5.0 MB | 5.2 MB | `262145 262143` | 4.6 | 3.6 | 1.0 | 4.0 | 2.7 | 1.3 |
| near_power_of_2_08 | 4.9 MB | 5.2 MB | `262145 262145` | 4.6 | 3.5 | 1.1 | 4.0 | 2.7 | 1.3 |
| near_power_of_2_04 | 4.8 MB | 5.0 MB | `262144 262144` | 4.6 | 3.3 | 1.3 | 4.0 | 2.7 | 1.3 |
| random_02 | 4.2 MB | 4.3 MB | `53336 382347` | 4.0 | 3.2 | 0.9 | 3.3 | 2.3 | 1.0 |
| example_00 | 22 B | 16 B | `4 5` | 1.4 | 0.8 | 0.6 | 0.8 | 0.5 | 0.3 |
| med_random_00 | 11.4 KB | 11.9 KB | `792 398` | 1.2 | 0.8 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_03 | 33 B | 21 B | `2 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| med_random_01 | 6.1 KB | 6.3 KB | `297 334` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_00 | 24 B | 10 B | `1 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| med_random_02 | 6.6 KB | 6.8 KB | `187 494` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_06 | 44 B | 33 B | `3 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_01 | 34 B | 20 B | `1 2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_05 | 54 B | 40 B | `2 3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_02 | 42 B | 32 B | `1 3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_08 | 63 B | 53 B | `3 3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_07 | 53 B | 41 B | `3 2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_04 | 44 B | 32 B | `2 2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
