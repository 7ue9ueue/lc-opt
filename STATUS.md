# Status

Updated: 2026-10-09

## Foundations

| Task | State | Notes |
|---|---|---|
| Machines | Done | GCP `lc-amd` (judge CPU) times, `lc-intel` profiles with `perf`, GitHub Actions times across CPUs. |
| Judge harness and CI auto-merge | Done | Matches the judge: QPoly 406478 16.6 ms (judged 16), matrix_product 53.4 ms (judged 53). |
| Judge submission (`tools/submit.py`) | Done | Submit as Aiyiyi. User logs in once; only a renewable token is kept, in macOS Keychain. Enforces 5 per problem. Fallback: user uploads by hand. |
| Round brief, issue template, `/work` | Built | `tools/prompt.md`, `.github/ISSUE_TEMPLATE/problem.md`, `.claude/commands/work.md`. Subagents in auto mode; not run yet. |
| `lib/io`: fast input and output | Done | Own code, no QgQ parts. EPYC 7B13: read 2.2 ns per uint32, bulk write 2.5 ns; many_aplusb 270 → 32 ms (record 23). See `lib/io/notes.md`. |
| `lib/ntt`: refactor the QPoly NTT | Not started | Reuse its kernels; rewrite for readability. |

## Problems

One GitHub issue per problem: https://github.com/7ue9ueue/lc-opt/issues
