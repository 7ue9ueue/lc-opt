# multivariate_convolution_cyclic

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 11.4.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.9.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| dim2_01 | 4.8 MB | 2.4 MB | `245309689 2` | 11.4 | 2.8 | 8.7 | 12.8 | 1.7 | 11.1 |
| dim1_00 | 4.9 MB | 2.5 MB | `778962101 1` | 11.1 | 2.7 | 8.4 | 11.8 | 1.8 | 10.0 |
| dim1_01 | 4.9 MB | 2.4 MB | `431735333 1` | 10.9 | 2.7 | 8.3 | 11.5 | 1.7 | 9.8 |
| dim1_02 | 4.9 MB | 2.5 MB | `854747833 1` | 10.8 | 2.7 | 8.1 | 11.4 | 1.8 | 9.7 |
| dim2_02 | 4.0 MB | 2.0 MB | `750771613 2` | 10.7 | 2.4 | 8.3 | 11.6 | 1.6 | 10.0 |
| dim2_00 | 4.5 MB | 2.3 MB | `168907411 2` | 10.4 | 2.7 | 7.7 | 10.7 | 1.7 | 9.0 |
| max_random_01 | 3.4 MB | 1.7 MB | `678341161 7` | 7.9 | 2.0 | 5.9 | 6.3 | 1.3 | 5.0 |
| twos_00 | 4.9 MB | 2.4 MB | `384558443 18` | 7.1 | 2.6 | 4.4 | 5.5 | 1.8 | 3.6 |
| threes_00 | 3.5 MB | 1.7 MB | `577279819 14` | 6.7 | 2.1 | 4.7 | 5.1 | 1.3 | 3.8 |
| threes_02 | 3.1 MB | 1.6 MB | `42367729 14` | 6.6 | 1.9 | 4.7 | 5.2 | 1.2 | 4.0 |
| max_random_00 | 2.8 MB | 1.4 MB | `609481321 7` | 6.4 | 1.9 | 4.5 | 5.5 | 1.2 | 4.3 |
| threes_01 | 2.6 MB | 1.3 MB | `932750587 13` | 5.3 | 1.7 | 3.6 | 4.2 | 1.1 | 3.1 |
| twos_01 | 2.5 MB | 1.2 MB | `820268453 17` | 4.1 | 1.7 | 2.4 | 3.1 | 1.0 | 2.0 |
| twos_02 | 989.2 KB | 494.5 KB | `4056077 16` | 2.5 | 1.1 | 1.5 | 1.9 | 0.7 | 1.2 |
| example_00 | 29 B | 14 B | `101 1` | 1.2 | 0.7 | 0.5 | 0.7 | 0.5 | 0.3 |
| small_00 | 54 B | 20 B | `613294831 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| example_01 | 36 B | 14 B | `13 2` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_01 | 132 B | 59 B | `850192621 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_03 | 64 B | 25 B | `30513667 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_04 | 94 B | 39 B | `769319849 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_02 | 150 B | 69 B | `914759903 1` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| k0_01 | 32 B | 10 B | `574343411 0` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| example_02 | 21 B | 6 B | `998244353 0` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| k0_00 | 33 B | 10 B | `648104887 0` | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
