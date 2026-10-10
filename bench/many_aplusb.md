# many_aplusb

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 17.7.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 15.2.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_max_00 | `T = 1000000` | 38.1 MB | 19.1 MB | 17.7 | 16.9 | 0.8 | 15.2 | 12.2 | 3.0 |
| max_random_01 | `T = 1000000` | 36.0 MB | 18.6 MB | 17.3 | 14.6 | 2.7 | 14.8 | 11.6 | 3.2 |
| max_random_00 | `T = 1000000` | 36.0 MB | 18.6 MB | 17.2 | 13.8 | 3.3 | 14.6 | 11.6 | 3.0 |
| digit_random_00 | `T = 1000000` | 20.0 MB | 10.6 MB | 16.8 | 8.9 | 7.9 | 12.8 | 6.7 | 6.1 |
| digit_random_01 | `T = 1000000` | 20.0 MB | 10.6 MB | 16.1 | 8.3 | 7.9 | 13.1 | 6.7 | 6.4 |
| random_01 | `T = 463046` | 16.7 MB | 8.6 MB | 8.8 | 6.8 | 2.0 | 7.4 | 5.4 | 2.1 |
| random_00 | `T = 389813` | 14.0 MB | 7.2 MB | 8.2 | 6.1 | 2.1 | 6.3 | 4.5 | 1.7 |
| all_zero_00 | `T = 1000000` | 3.8 MB | 1.9 MB | 6.4 | 2.1 | 4.3 | 4.9 | 1.5 | 3.4 |
| example_00 | `T = 3` | 52 B | 25 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
