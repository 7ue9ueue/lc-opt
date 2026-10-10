# many_aplusb

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 17.7.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 15.2.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| all_max_00 | 38.1 MB | 19.1 MB | `1000000` | 17.7 | 16.9 | 0.8 | 15.2 | 12.2 | 3.0 |
| max_random_01 | 36.0 MB | 18.6 MB | `1000000` | 17.3 | 14.6 | 2.7 | 14.8 | 11.6 | 3.2 |
| max_random_00 | 36.0 MB | 18.6 MB | `1000000` | 17.2 | 13.8 | 3.3 | 14.6 | 11.6 | 3.0 |
| digit_random_00 | 20.0 MB | 10.6 MB | `1000000` | 16.8 | 8.9 | 7.9 | 12.8 | 6.7 | 6.1 |
| digit_random_01 | 20.0 MB | 10.6 MB | `1000000` | 16.1 | 8.3 | 7.9 | 13.1 | 6.7 | 6.4 |
| random_01 | 16.7 MB | 8.6 MB | `463046` | 8.8 | 6.8 | 2.0 | 7.4 | 5.4 | 2.1 |
| random_00 | 14.0 MB | 7.2 MB | `389813` | 8.2 | 6.1 | 2.1 | 6.3 | 4.5 | 1.7 |
| all_zero_00 | 3.8 MB | 1.9 MB | `1000000` | 6.4 | 2.1 | 4.3 | 4.9 | 1.5 | 3.4 |
| example_00 | 52 B | 25 B | `3` | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
