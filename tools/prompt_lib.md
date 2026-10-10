You are working on the shared module `{path}`, tracked in issue #{issue}.
You are unattended: the user is offline, so do not ask questions. Decide and act.

1. Read `AGENTS.md`, issue #{issue} with its comments, and `{path}/notes.md`.
2. This round: do the task in the issue. A change to `lib/` re-times every problem in CI, and
   noise alone can fail the no-slowdown check. Prefer adding functions over changing existing hot paths.
   Do not repeat a logged attempt without a new reason.
3. Run the module's tests, then `tools/judge.py test` on every problem that uses the module.
   Time on `lc-amd`; profile on `lc-intel`.
4. Log the attempt in `{path}/notes.md`. Open a pull request from a branch `agent/lib-<topic>` and run
   `gh pr merge --auto --squash`. Wait until it merges or CI fails; if CI fails, fix it or close the pull request.
5. Finish with one comment on issue #{issue}: what you tried, the numbers, and a last line that is exactly
   `Result: gain` or `Result: no gain`.

If something only the user can fix stops you, comment why on issue #{issue} and add the `blocked` label.
