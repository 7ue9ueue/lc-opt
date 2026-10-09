# aplusb

Read two integers, print their sum. Baseline only; used to test the CI pipeline.

## Log

- 2026-10-09, claude: baseline `scanf`/`printf`.
- 2026-10-09, claude: submitted the baseline, [409068](https://judge.yosupo.jp/submission/409068): AC, 10 ms (1/5).
- 2026-10-09, claude: submitted `tools/isa_probe.cpp`, [409083](https://judge.yosupo.jp/submission/409083): AC, 10 ms (2/5).
  All 24 instruction-set checks hold on the judge: `znver3`, BMI1/2, ADX, PCLMUL, VPCLMULQDQ, AES, VAES; AVX-512 faults.
  Not a CI-checked `main.cpp`; it passed all 12 official tests on `lc-amd` first.
- 2026-10-09, claude: `lib/io` instead of `scanf`/`printf`; EPYC 7B13, 41 rounds: ratio 0.99 (1.14 vs 1.15 ms).
