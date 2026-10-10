# multivariate_convolution

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 13.8. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 11.8. Older `main.cpp` than the current one.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| twos_00 | 4.9 MB | 2.5 MB | `18` | 13.7 | 2.7 | 10.9 | 11.8 | 1.7 | 10.1 |
| threes_00 | 3.5 MB | 1.8 MB | `14` | 12.9 | 2.1 | 10.8 | 11.4 | 1.4 | 10.0 |
| max_random_01 | 3.4 MB | 1.7 MB | `7` | 11.5 | 2.1 | 9.4 | 11.4 | 1.3 | 10.1 |
| max_random_00 | 2.9 MB | 1.4 MB | `7` | 10.4 | 1.9 | 8.5 | 10.4 | 1.2 | 9.2 |
| threes_01 | 2.6 MB | 1.3 MB | `13` | 10.3 | 1.8 | 8.5 | 9.0 | 1.1 | 7.8 |
| dim2_01 | 4.9 MB | 2.5 MB | `2` | 9.3 | 2.7 | 6.6 | 9.7 | 1.7 | 8.0 |
| dim2_00 | 4.8 MB | 2.4 MB | `2` | 9.3 | 2.7 | 6.6 | 9.6 | 1.8 | 7.8 |
| twos_01 | 2.5 MB | 1.2 MB | `17` | 7.0 | 1.7 | 5.3 | 6.0 | 1.0 | 5.0 |
| dim1_00 | 4.9 MB | 2.5 MB | `1` | 7.0 | 2.7 | 4.2 | 6.2 | 1.6 | 4.5 |
| dim1_01 | 4.9 MB | 2.5 MB | `1` | 6.8 | 2.7 | 4.0 | 6.0 | 1.7 | 4.3 |
| example_00 | 25 B | 15 B | `1` | 1.2 | 0.7 | 0.5 | 0.7 | 0.5 | 0.3 |
| example_01 | 33 B | 18 B | `2` | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| example_02 | 11 B | 6 B | `0` | 1.2 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_00 | 11.6 KB | 5.8 KB | `2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| small_01 | 18.3 KB | 9.1 KB | `3` | 1.1 | 0.7 | 0.4 | 0.6 | 0.5 | 0.2 |
| k0_00 | 23 B | 10 B | `0` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| k0_01 | 23 B | 10 B | `0` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
