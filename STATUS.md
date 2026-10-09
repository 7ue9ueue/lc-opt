# Status

Updated: 2026-10-09

## Foundations

| Task | State | Notes |
|---|---|---|
| Machines | Done | GCP `lc-amd` (judge CPU) times, `lc-intel` profiles with `perf`, GitHub Actions times across CPUs. |
| Judge harness and CI auto-merge | Done | Matches the judge: QPoly 406478 16.6 ms (judged 16), matrix_product 53.4 ms (judged 53). |
| Judge submission (`tools/submit.py`) | Done | Submit as Aiyiyi. User logs in once; only a renewable token is kept, in macOS Keychain. Enforces 5 per problem. Fallback: user uploads by hand. |
| Standing prompt, issue template, dispatcher | Planned | Headless Claude Code, 5–10 agents, one per problem. |
| `lib/io`: fast input and output | Done | Own code, no QgQ parts. EPYC 7B13: read 2.2 ns per uint32, bulk write 2.5 ns; many_aplusb 270 → 32 ms (record 23). See `lib/io/notes.md`. |
| `lib/ntt`: refactor the QPoly NTT | Not started | Reuse its kernels; rewrite for readability. |

## Problems

Pass: 0 research, 1 high level, 2 SIMD, 3 assembly. Times in ms; `J` judged, `E` estimated.

| Problem | Pass | Best | Record | Next |
|---|---|---|---|---|
| aplusb | 1 | 10 J | — | Pipeline test only. |
