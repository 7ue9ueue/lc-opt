# multivariate_convolution_cyclic

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 11.4.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.9.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dim2_01 | `P K = 245309689 2` | 4.8 MB | 2.4 MB | 11.4 | 2.8 | 8.7 | 12.8 | 1.7 | 11.1 |
| dim1_00 | `P K = 778962101 1` | 4.9 MB | 2.5 MB | 11.1 | 2.7 | 8.4 | 11.8 | 1.8 | 10.0 |
| dim1_01 | `P K = 431735333 1` | 4.9 MB | 2.4 MB | 10.9 | 2.7 | 8.3 | 11.5 | 1.7 | 9.8 |
| dim1_02 | `P K = 854747833 1` | 4.9 MB | 2.5 MB | 10.8 | 2.7 | 8.1 | 11.4 | 1.8 | 9.7 |
| dim2_02 | `P K = 750771613 2` | 4.0 MB | 2.0 MB | 10.7 | 2.4 | 8.3 | 11.6 | 1.6 | 10.0 |
| dim2_00 | `P K = 168907411 2` | 4.5 MB | 2.3 MB | 10.4 | 2.7 | 7.7 | 10.7 | 1.7 | 9.0 |
| max_random_01 | `P K = 678341161 7` | 3.4 MB | 1.7 MB | 7.9 | 2.0 | 5.9 | 6.3 | 1.3 | 5.0 |
| twos_00 | `P K = 384558443 18` | 4.9 MB | 2.4 MB | 7.1 | 2.6 | 4.4 | 5.5 | 1.8 | 3.6 |
| threes_00 | `P K = 577279819 14` | 3.5 MB | 1.7 MB | 6.7 | 2.1 | 4.7 | 5.1 | 1.3 | 3.8 |
| threes_02 | `P K = 42367729 14` | 3.1 MB | 1.6 MB | 6.6 | 1.9 | 4.7 | 5.2 | 1.2 | 4.0 |
| max_random_00 | `P K = 609481321 7` | 2.8 MB | 1.4 MB | 6.4 | 1.9 | 4.5 | 5.5 | 1.2 | 4.3 |
| threes_01 | `P K = 932750587 13` | 2.6 MB | 1.3 MB | 5.3 | 1.7 | 3.6 | 4.2 | 1.1 | 3.1 |
| twos_01 | `P K = 820268453 17` | 2.5 MB | 1.2 MB | 4.1 | 1.7 | 2.4 | 3.1 | 1.0 | 2.0 |
| twos_02 | `P K = 4056077 16` | 989.2 KB | 494.5 KB | 2.5 | 1.1 | 1.5 | 1.9 | 0.7 | 1.2 |
| example_00 | `P K = 101 1` | 29 B | 14 B | 1.2 | 0.7 | 0.5 | 0.7 | 0.5 | 0.3 |
| small_00 | `P K = 613294831 1` | 54 B | 20 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| example_01 | `P K = 13 2` | 36 B | 14 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| small_01 | `P K = 850192621 1` | 132 B | 59 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_03 | `P K = 30513667 1` | 64 B | 25 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_04 | `P K = 769319849 1` | 94 B | 39 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| small_02 | `P K = 914759903 1` | 150 B | 69 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.3 |
| k0_01 | `P K = 574343411 0` | 32 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
| example_02 | `P K = 998244353 0` | 21 B | 6 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.3 |
| k0_00 | `P K = 648104887 0` | 33 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.5 | 0.2 |
