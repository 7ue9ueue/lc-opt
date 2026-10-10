---
description: Run optimization rounds on ready problem issues with subagents, one per problem
argument-hint: "[agents at once, default 5]"
---
Read `AGENTS.md`. Work through the open GitHub issues labeled `ready`, fewest rounds first, then oldest, with up to
$ARGUMENTS subagents at once (5 if no number is given). Never two subagents on the same problem.
Start an issue only after every issue it lists under `After:` is closed.

For each round:
1. Relabel the issue from `ready` to `running`.
2. Start a subagent in its own git worktree. Brief it with `tools/prompt.md`, filling in `{problem}` (the issue
   title), `{issue}` (its number) and `{path}` (from the issue's `Folder:` line).
   For an issue labeled `lib`, use `tools/prompt_lib.md` instead.
3. When it returns, read the issue's newest `Result:` line. If the round left none, comment what happened,
   ending with `Result: no gain`.
4. Relabel the issue: leave it `blocked` if the agent asked for the user; mark it `done` and close it after
   5 rounds or 2 `no gain` results in a row; otherwise back to `ready`.

Keep starting rounds until no issue is `ready`, then report each problem's result in a few lines.

To add problems, open an issue per problem from `.github/ISSUE_TEMPLATE/problem.md` with label `ready`.
Find the folder by the problem's directory in library-checker-problems (`tools/cases.py` caches a checkout),
and the record as a time only, from `https://v3.api.judge.yosupo.jp/submissions?problem=<name>&status=AC&order=%2Btime&limit=1`.
