# mul_mod2n_convolution

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 20.6.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 19.3.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ans_zero_00 | `N = 20` | 19.9 MB | 2.0 MB | 20.4 | 4.5 | 15.8 | 18.7 | 3.4 | 15.3 |
| large_00 | `N = 20` | 19.8 MB | 9.9 MB | 20.2 | 8.1 | 12.1 | 18.3 | 5.9 | 12.4 |
| n_equals_20 | `N = 20` | 19.8 MB | 9.9 MB | 20.2 | 7.8 | 12.4 | 18.8 | 5.7 | 13.1 |
| large_01 | `N = 20` | 19.8 MB | 9.9 MB | 20.1 | 7.5 | 12.5 | 18.3 | 6.2 | 12.1 |
| large_02 | `N = 19` | 9.9 MB | 4.9 MB | 11.0 | 4.4 | 6.6 | 9.9 | 3.4 | 6.5 |
| n_equals_19 | `N = 19` | 9.9 MB | 4.9 MB | 11.0 | 4.3 | 6.7 | 9.8 | 3.2 | 6.6 |
| ans_zero_01 | `N = 19` | 9.9 MB | 1.0 MB | 10.9 | 2.7 | 8.3 | 9.6 | 1.8 | 7.8 |
| large_03 | `N = 19` | 9.9 MB | 4.9 MB | 10.7 | 4.0 | 6.7 | 9.8 | 3.3 | 6.5 |
| large_04 | `N = 18` | 4.9 MB | 2.5 MB | 5.9 | 2.5 | 3.4 | 5.3 | 1.7 | 3.5 |
| large_05 | `N = 18` | 4.9 MB | 2.5 MB | 5.8 | 2.4 | 3.4 | 5.3 | 1.8 | 3.5 |
| n_equals_18 | `N = 18` | 4.9 MB | 2.5 MB | 5.8 | 2.4 | 3.4 | 5.3 | 1.8 | 3.5 |
| n_equals_17 | `N = 17` | 2.5 MB | 1.2 MB | 3.5 | 1.5 | 1.9 | 3.0 | 1.1 | 2.0 |
| n_equals_16 | `N = 16` | 1.2 MB | 632.9 KB | 2.4 | 1.1 | 1.2 | 2.0 | 0.8 | 1.2 |
| n_equals_15 | `N = 15` | 632.8 KB | 316.5 KB | 1.8 | 0.9 | 0.8 | 1.4 | 0.6 | 0.8 |
| all_zero_03 | `N = 15` | 128.0 KB | 64.0 KB | 1.7 | 0.8 | 1.0 | 1.2 | 0.5 | 0.7 |
| n_equals_14 | `N = 14` | 316.4 KB | 158.2 KB | 1.5 | 0.8 | 0.6 | 1.1 | 0.5 | 0.6 |
| n_equals_13 | `N = 13` | 158.2 KB | 79.1 KB | 1.4 | 0.8 | 0.6 | 0.9 | 0.5 | 0.4 |
| all_zero_00 | `N = 0` | 6 B | 2 B | 1.3 | 0.7 | 0.6 | 0.7 | 0.5 | 0.3 |
| n_equals_12 | `N = 12` | 79.1 KB | 39.6 KB | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| n_equals_11 | `N = 11` | 39.6 KB | 19.8 KB | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| small_00 | `N = 0` | 22 B | 10 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| n_equals_10 | `N = 10` | 19.8 KB | 9.9 KB | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| all_zero_02 | `N = 3` | 34 B | 16 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| all_zero_01 | `N = 1` | 10 B | 4 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| example_00 | `N = 2` | 18 B | 13 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| n_equals_04 | `N = 4` | 318 B | 158 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| n_equals_09 | `N = 9` | 9.9 KB | 4.9 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| n_equals_01 | `N = 1` | 41 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| n_equals_00 | `N = 0` | 22 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| n_equals_07 | `N = 7` | 2.5 KB | 1.2 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| n_equals_03 | `N = 3` | 158 B | 77 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| example_01 | `N = 0` | 22 B | 2 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_05 | `N = 1` | 42 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_06 | `N = 2` | 81 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_09 | `N = 3` | 162 B | 79 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_02 | `N = 0` | 22 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| n_equals_06 | `N = 6` | 1.2 KB | 634 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_03 | `N = 1` | 42 B | 19 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_11 | `N = 3` | 160 B | 75 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| n_equals_08 | `N = 8` | 4.9 KB | 2.5 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_07 | `N = 2` | 82 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_01 | `N = 0` | 22 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| n_equals_05 | `N = 5` | 634 B | 318 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_04 | `N = 1` | 42 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_10 | `N = 3` | 162 B | 80 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| n_equals_02 | `N = 2` | 80 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_08 | `N = 2` | 79 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
