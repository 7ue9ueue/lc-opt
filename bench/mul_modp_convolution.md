# mul_modp_convolution

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.3.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.5.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p_max_02 | `P = 524287` | 9.9 MB | 4.9 MB | 12.3 | 4.7 | 7.6 | 12.3 | 3.1 | 9.3 |
| p_max_00 | `P = 524287` | 9.9 MB | 4.9 MB | 12.2 | 4.6 | 7.6 | 12.3 | 3.1 | 9.2 |
| large_05 | `P = 518737` | 9.8 MB | 4.9 MB | 12.1 | 4.6 | 7.6 | 12.3 | 3.1 | 9.2 |
| p_max_01 | `P = 524287` | 9.9 MB | 4.9 MB | 12.0 | 4.6 | 7.4 | 12.3 | 3.0 | 9.4 |
| p_max_03 | `P = 524287` | 9.9 MB | 4.9 MB | 12.0 | 4.5 | 7.4 | 12.3 | 3.0 | 9.3 |
| large_04 | `P = 209639` | 4.0 MB | 2.0 MB | 7.2 | 2.3 | 4.9 | 6.3 | 1.4 | 4.8 |
| large_03 | `P = 128473` | 2.4 MB | 1.2 MB | 3.7 | 1.7 | 2.0 | 3.2 | 1.1 | 2.1 |
| large_02 | `P = 59453` | 1.1 MB | 574.0 KB | 2.7 | 1.2 | 1.6 | 2.1 | 0.7 | 1.3 |
| large_01 | `P = 30323` | 585.7 KB | 292.9 KB | 1.7 | 0.9 | 0.8 | 1.3 | 0.6 | 0.6 |
| large_00 | `P = 14737` | 284.7 KB | 142.3 KB | 1.6 | 0.8 | 0.7 | 1.1 | 0.5 | 0.5 |
| all_zero_00 | `P = 2` | 10 B | 4 B | 1.3 | 0.7 | 0.6 | 0.7 | 0.5 | 0.2 |
| medium_09 | `P = 3989` | 77.1 KB | 38.5 KB | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| medium_11 | `P = 4999` | 96.6 KB | 48.3 KB | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| medium_10 | `P = 6653` | 128.5 KB | 64.2 KB | 1.2 | 0.8 | 0.5 | 0.8 | 0.5 | 0.3 |
| medium_08 | `P = 2539` | 49.1 KB | 24.5 KB | 1.2 | 0.7 | 0.5 | 0.8 | 0.5 | 0.3 |
| medium_07 | `P = 1489` | 28.8 KB | 14.4 KB | 1.2 | 0.7 | 0.5 | 0.7 | 0.5 | 0.3 |
| medium_05 | `P = 947` | 18.3 KB | 9.1 KB | 1.2 | 0.7 | 0.5 | 0.7 | 0.5 | 0.3 |
| medium_04 | `P = 541` | 10.5 KB | 5.2 KB | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| medium_06 | `P = 1129` | 21.8 KB | 10.9 KB | 1.2 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| medium_00 | `P = 191` | 3.7 KB | 1.8 KB | 1.1 | 0.7 | 0.4 | 0.8 | 0.5 | 0.3 |
| small_00 | `P = 2` | 42 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| all_zero_01 | `P = 3` | 14 B | 6 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_06 | `P = 23` | 460 B | 228 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_04 | `P = 11` | 222 B | 109 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| medium_03 | `P = 479` | 9.3 KB | 4.6 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| example_00 | `P = 5` | 22 B | 16 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_01 | `P = 3` | 61 B | 30 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| medium_02 | `P = 317` | 6.1 KB | 3.1 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_02 | `P = 5` | 102 B | 50 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_08 | `P = 41` | 814 B | 401 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| medium_01 | `P = 181` | 3.5 KB | 1.7 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| all_zero_02 | `P = 5` | 22 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| all_zero_03 | `P = 7` | 30 B | 14 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| example_01 | `P = 2` | 26 B | 4 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| small_03 | `P = 5` | 101 B | 50 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_07 | `P = 19` | 381 B | 189 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_05 | `P = 13` | 258 B | 129 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_09 | `P = 37` | 730 B | 366 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_10 | `P = 107` | 2.1 KB | 1.0 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_11 | `P = 109` | 2.1 KB | 1.1 KB | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
