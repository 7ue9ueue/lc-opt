You are working on Library Checker problem `{problem}`, tracked in issue #{issue}.
You are unattended: the user is offline, so do not ask questions. Decide and act.

1. Read `AGENTS.md`, issue #{issue} with its comments, and `{path}/notes.md` if it exists.
2. This round: if `{path}/main.cpp` does not exist, write a correct solution. Otherwise make it faster.
   Choose your own approach. Do not repeat a logged attempt without a new reason.
   Do not edit `lib/` unless issue #{issue} says you own that module; put problem-specific code in
   `{path}/solution.cpp`. At most one open pull request may touch `lib/` at a time.
3. Check your work with `tools/judge.py` on a VM: `lc-amd` for timing, `lc-intel` for `perf`.
4. Log the attempt in `{path}/notes.md`. Open a pull request from branch `agent/{problem}` and run
   `gh pr merge --auto --squash`. Wait until it merges or CI fails; if CI fails, fix it or close the pull request.
5. If the merged `main.cpp` beats the best judged time in `notes.md`, submit it with `tools/submit.py`
   and log the result in `notes.md` through another pull request.
6. Finish with one comment on issue #{issue}: what you tried, the numbers, and a last line that is exactly
   `Result: gain` or `Result: no gain`.

If something only the user can fix stops you, comment why on issue #{issue} and add the `blocked` label.
