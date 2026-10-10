# bitwise_and_convolution

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 12.5. Older `main.cpp` than the current one.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 10.2. Older `main.cpp` than the current one.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| max_random_00 | 19.8 MB | 9.9 MB | `20` | 12.5 | 8.4 | 4.1 | 9.8 | 5.8 | 4.0 |
| max_random_01 | 19.8 MB | 9.9 MB | `20` | 12.3 | 8.4 | 4.0 | 9.9 | 5.9 | 4.0 |
| max_random_02 | 19.8 MB | 9.9 MB | `20` | 12.3 | 8.3 | 4.0 | 9.8 | 5.3 | 4.5 |
| random_02 | 4.9 MB | 2.5 MB | `18` | 3.9 | 2.6 | 1.3 | 3.0 | 1.7 | 1.3 |
| random_01 | 1.2 MB | 632.7 KB | `16` | 2.0 | 1.2 | 0.8 | 1.4 | 0.7 | 0.6 |
| random_00 | 632.9 KB | 316.5 KB | `15` | 1.7 | 1.0 | 0.8 | 1.1 | 0.6 | 0.5 |
| example_00 | 41 B | 32 B | `3` | 1.2 | 0.7 | 0.4 | 0.6 | 0.5 | 0.2 |
| small_01 | 1.2 KB | 631 B | `6` | 1.1 | 0.7 | 0.3 | 0.7 | 0.5 | 0.2 |
| small_02 | 4.9 KB | 2.5 KB | `8` | 1.1 | 0.7 | 0.3 | 0.6 | 0.5 | 0.1 |
| small_00 | 640 B | 319 B | `5` | 1.1 | 0.7 | 0.3 | 0.7 | 0.5 | 0.2 |
| tiny_00 | 22 B | 10 B | `0` | 1.0 | 0.7 | 0.3 | 0.6 | 0.5 | 0.1 |
| tiny_01 | 41 B | 19 B | `1` | 1.0 | 0.7 | 0.3 | 0.6 | 0.5 | 0.1 |
| tiny_02 | 80 B | 40 B | `2` | 1.0 | 0.8 | 0.3 | 0.6 | 0.5 | 0.1 |
