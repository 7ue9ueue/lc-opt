# aplusb

Every official test, median over rounds, ms. Floor: start, map the input, write an output of the expected size, nothing else (`tools/floor.c`). Compute: total minus floor, so parsing, work and formatting.

- AMD: AMD EPYC 7B13, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=native -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 1.2.
- Intel: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, `g++ -O2 -std=c++23 -DEVAL -DONLINE_JUDGE -march=x86-64-v3 -madx -mpclmul -mvpclmulqdq -maes -mvaes -o main main.cpp`, commit 3740c29, 5 rounds, 2026-10-10. Score 0.7.

| Test | Input | Output | First line | AMD | AMD floor | AMD compute | Intel | Intel floor | Intel compute |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| example_00 | 10 B | 5 B | `1234 5678` | 1.2 | 0.7 | 0.4 | 0.6 | 0.5 | 0.2 |
| random_05 | 20 B | 10 B | `173330281 220603612` | 1.2 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| random_00 | 20 B | 10 B | `192279220 156648746` | 1.2 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_04 | 20 B | 11 B | `729561619 415335212` | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_02 | 20 B | 11 B | `682152023 451794314` | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_06 | 19 B | 10 B | `841413509 58432763` | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_09 | 20 B | 11 B | `907649120 290651129` | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_03 | 20 B | 11 B | `627477696 504915182` | 1.1 | 0.7 | 0.4 | 0.6 | 0.5 | 0.1 |
| random_01 | 20 B | 10 B | `264704197 120999146` | 1.1 | 0.7 | 0.4 | 0.7 | 0.4 | 0.2 |
| random_08 | 20 B | 10 B | `118232767 222490630` | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| random_07 | 20 B | 10 B | `251229786 256388306` | 1.1 | 0.7 | 0.4 | 0.6 | 0.4 | 0.2 |
| example_01 | 22 B | 11 B | `1000000000 1000000000` | 1.1 | 0.7 | 0.4 | 0.6 | 0.5 | 0.2 |
