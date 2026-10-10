# gcd_convolution

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 15.4.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 14.0.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| max_random_00 | 18.9 MB | 9.4 MB | `1000000` | 15.4 | 8.0 | 7.4 | 13.5 | 6.0 | 7.5 |
| max_random_01 | 18.9 MB | 9.4 MB | `1000000` | 14.5 | 7.9 | 6.6 | 13.3 | 5.9 | 7.3 |
| near_prime_00 | 18.9 MB | 9.4 MB | `999982` | 14.0 | 7.6 | 6.5 | 13.5 | 6.2 | 7.3 |
| near_prime_squared_02 | 18.7 MB | 9.4 MB | `994010` | 14.0 | 7.7 | 6.3 | 13.1 | 6.2 | 6.9 |
| near_prime_01 | 18.9 MB | 9.4 MB | `999983` | 13.9 | 7.0 | 6.8 | 13.3 | 6.2 | 7.1 |
| near_prime_02 | 18.9 MB | 9.4 MB | `999984` | 13.9 | 7.1 | 6.8 | 13.6 | 5.9 | 7.7 |
| near_prime_squared_01 | 18.7 MB | 9.4 MB | `994009` | 13.8 | 7.7 | 6.1 | 13.3 | 6.0 | 7.2 |
| near_prime_squared_00 | 18.7 MB | 9.4 MB | `994008` | 13.7 | 7.2 | 6.4 | 13.5 | 6.5 | 7.0 |
| random_02 | 10.9 MB | 5.4 MB | `577624` | 8.5 | 4.6 | 3.9 | 7.8 | 3.6 | 4.2 |
| random_01 | 8.7 MB | 4.4 MB | `463046` | 7.1 | 3.9 | 3.2 | 6.1 | 3.0 | 3.1 |
| random_00 | 7.4 MB | 3.7 MB | `389813` | 6.0 | 3.2 | 2.9 | 5.5 | 2.7 | 2.8 |
| all_zero_00 | 26 B | 12 B | `6` | 1.3 | 0.7 | 0.6 | 0.8 | 0.5 | 0.3 |
| example_00 | 26 B | 18 B | `6` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_00 | 22 B | 10 B | `1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_05 | 121 B | 58 B | `6` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_01 | 41 B | 19 B | `2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_03 | 82 B | 39 B | `4` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_11 | 240 B | 120 B | `12` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_02 | 60 B | 30 B | `3` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_06 | 140 B | 70 B | `7` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_08 | 179 B | 89 B | `9` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_14 | 301 B | 149 B | `15` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_10 | 222 B | 108 B | `11` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_13 | 283 B | 137 B | `14` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_07 | 159 B | 80 B | `8` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_09 | 203 B | 100 B | `10` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_12 | 261 B | 129 B | `13` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_15 | 320 B | 158 B | `16` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_04 | 102 B | 49 B | `5` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
