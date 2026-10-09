# Status

Updated: 2026-10-09

## Foundations

| Task | State | Notes |
|---|---|---|
| Machines | Done | GCP `lc-amd` (judge CPU) times, `lc-intel` profiles with `perf`, GitHub Actions times across CPUs. |
| Judge harness and CI auto-merge | Done | Matches the judge: QPoly 406478 16.6 ms (judged 16), matrix_product 53.4 ms (judged 53). |
| `lib/io`: fast input and output | Next | First task. Independent of the rest. |
| `lib/ntt`: refactor the QPoly NTT | Not started | Reuse its kernels; rewrite for readability. |

## Problems

Pass: 0 research, 1 high level, 2 SIMD, 3 assembly. Times in ms; `J` judged, `E` estimated.

| Problem | Pass | Best | Record | Next |
|---|---|---|---|---|
| aplusb | 1 | — | — | Pipeline test only. |
