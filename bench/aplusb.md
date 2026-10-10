# aplusb

Every official test, median over rounds, ms. Size: the first input line, named as in the official input format. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 1.2.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 0.7.

| Test | Size | Input | Output | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| example_00 | `A B = 1234 5678` | 10 B | 5 B | 1.2 | 0.7 | 0.4 | 0.6 | 0.5 | 0.2 |
| random_05 | `A B = 173330281 220603612` | 20 B | 10 B | 1.2 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| random_00 | `A B = 192279220 156648746` | 20 B | 10 B | 1.2 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_04 | `A B = 729561619 415335212` | 20 B | 11 B | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_02 | `A B = 682152023 451794314` | 20 B | 11 B | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_06 | `A B = 841413509 58432763` | 19 B | 10 B | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_09 | `A B = 907649120 290651129` | 20 B | 11 B | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_03 | `A B = 627477696 504915182` | 20 B | 11 B | 1.1 | 0.7 | 0.4 | 0.6 | 0.5 | 0.1 |
| random_01 | `A B = 264704197 120999146` | 20 B | 10 B | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| random_08 | `A B = 118232767 222490630` | 20 B | 10 B | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_07 | `A B = 251229786 256388306` | 20 B | 10 B | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| example_01 | `A B = 1000000000 1000000000` | 22 B | 11 B | 1.1 | 0.7 | 0.4 | 0.6 | 0.5 | 0.2 |
