# convolution_mod

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 13.2. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.2. Older `main.cpp` than the current one.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_random_00 | `N M = 524288 524288` | 9.9 MB | 9.9 MB | 12.6 | 6.2 | 6.4 | 11.7 | 4.5 | 7.2 |
| max_ans_zero_00 | `N M = 524288 524288` | 9.9 MB | 9.9 MB | 12.3 | 6.1 | 6.2 | 11.8 | 4.6 | 7.2 |
| fft_killer_07 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.3 | 6.1 | 6.1 | 11.6 | 4.5 | 7.1 |
| max_random_01 | `N M = 524288 524288` | 9.9 MB | 9.9 MB | 12.2 | 6.1 | 6.2 | 11.9 | 4.6 | 7.2 |
| all_same_01 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.2 | 6.4 | 5.7 | 11.6 | 4.6 | 6.9 |
| fft_killer_04 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.2 | 6.2 | 6.0 | 11.6 | 4.8 | 6.9 |
| fft_killer_08 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.1 | 6.1 | 6.0 | 11.6 | 4.5 | 7.0 |
| fft_killer_01 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.1 | 6.6 | 5.5 | 11.7 | 4.7 | 7.0 |
| all_same_03 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.1 | 6.3 | 5.8 | 11.7 | 4.8 | 6.9 |
| fft_killer_02 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.1 | 6.3 | 5.8 | 11.5 | 4.6 | 6.9 |
| fft_killer_05 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.1 | 6.0 | 6.1 | 11.7 | 4.6 | 7.1 |
| fft_killer_06 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.1 | 6.1 | 6.0 | 11.9 | 4.8 | 7.1 |
| fft_killer_09 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.1 | 6.0 | 6.0 | 11.5 | 4.5 | 7.0 |
| fft_killer_00 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 12.0 | 6.1 | 5.9 | 11.5 | 4.6 | 6.9 |
| fft_killer_03 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 11.9 | 6.1 | 5.9 | 11.5 | 4.7 | 6.8 |
| all_same_02 | `N M = 524288 524288` | 10.0 MB | 9.9 MB | 11.9 | 6.4 | 5.5 | 11.7 | 4.7 | 7.0 |
| all_same_00 | `N M = 524288 524288` | 2.0 MB | 6.8 MB | 11.5 | 3.7 | 7.8 | 11.2 | 3.0 | 8.3 |
| random_01 | `N M = 463046 412907` | 8.3 MB | 8.3 MB | 11.1 | 5.2 | 5.9 | 11.1 | 4.0 | 7.1 |
| random_00 | `N M = 389813 410923` | 7.6 MB | 7.6 MB | 10.7 | 4.8 | 5.9 | 10.5 | 3.5 | 7.0 |
| small_and_large_01 | `N M = 1000 524288` | 5.0 MB | 5.0 MB | 9.0 | 3.4 | 5.6 | 9.3 | 2.5 | 6.7 |
| small_and_large_03 | `N M = 524288 1000` | 5.0 MB | 5.0 MB | 9.0 | 3.4 | 5.6 | 9.4 | 2.6 | 6.8 |
| small_and_large_02 | `N M = 524288 100` | 4.9 MB | 4.9 MB | 9.0 | 3.4 | 5.6 | 9.3 | 2.5 | 6.7 |
| small_and_large_00 | `N M = 100 524288` | 4.9 MB | 4.9 MB | 8.9 | 3.4 | 5.6 | 9.4 | 2.5 | 6.8 |
| random_02 | `N M = 53336 382347` | 4.1 MB | 4.1 MB | 6.2 | 3.0 | 3.2 | 5.8 | 2.2 | 3.6 |
| medium_00 | `N M = 1323 9953` | 108.9 KB | 108.9 KB | 1.4 | 0.8 | 0.6 | 0.9 | 0.5 | 0.4 |
| medium_all_zero_00 | `N M = 1323 9953` | 22.0 KB | 22.0 KB | 1.3 | 0.8 | 0.6 | 0.9 | 0.5 | 0.4 |
| medium_01 | `N M = 4294 3307` | 73.5 KB | 73.4 KB | 1.3 | 0.8 | 0.5 | 0.8 | 0.5 | 0.3 |
| medium_02 | `N M = 4184 5515` | 93.7 KB | 93.6 KB | 1.2 | 0.8 | 0.5 | 0.8 | 0.5 | 0.4 |
| medium_pre_suf_zero_01 | `N M = 1000 1000` | 19.3 KB | 19.3 KB | 1.2 | 0.7 | 0.4 | 0.8 | 0.5 | 0.3 |
| example_00 | `N M = 4 5` | 22 B | 23 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| medium_pre_suf_zero_04 | `N M = 1000 1000` | 19.3 KB | 19.3 KB | 1.2 | 0.7 | 0.4 | 0.8 | 0.5 | 0.3 |
| medium_pre_suf_zero_02 | `N M = 1000 1000` | 3.9 KB | 3.9 KB | 1.2 | 0.7 | 0.4 | 0.8 | 0.5 | 0.3 |
| medium_pre_suf_zero_00 | `N M = 1000 1000` | 19.3 KB | 19.3 KB | 1.2 | 0.8 | 0.4 | 0.8 | 0.5 | 0.3 |
| medium_pre_suf_zero_03 | `N M = 1000 1000` | 3.9 KB | 3.9 KB | 1.1 | 0.7 | 0.4 | 0.8 | 0.4 | 0.3 |
| signed_overflow_00 | `N M = 38 38` | 766 B | 622 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_08 | `N M = 3 1` | 44 B | 30 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| example_01 | `N M = 1 1` | 22 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| unsigned_overflow_00 | `N M = 18 18` | 366 B | 101 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_09 | `N M = 3 2` | 54 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_14 | `N M = 4 3` | 72 B | 59 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_00 | `N M = 1 1` | 24 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_01 | `N M = 1 2` | 34 B | 20 B | 1.1 | 0.8 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_11 | `N M = 3 4` | 73 B | 60 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_02 | `N M = 1 3` | 42 B | 29 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_15 | `N M = 4 4` | 83 B | 67 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_04 | `N M = 2 1` | 34 B | 19 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_07 | `N M = 2 4` | 64 B | 50 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_13 | `N M = 4 2` | 64 B | 49 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_10 | `N M = 3 3` | 64 B | 48 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_05 | `N M = 2 2` | 44 B | 30 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_06 | `N M = 2 3` | 53 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_12 | `N M = 4 1` | 53 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_03 | `N M = 1 4` | 54 B | 40 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
