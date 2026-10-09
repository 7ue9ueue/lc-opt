---
name: Problem
about: One Library Checker problem, worked on in rounds by agents
title: "problem_name"
labels: ready
---
Problem: https://judge.yosupo.jp/problem/{problem}
Folder: `{path}`
Record when opened: {record}

Agents work on this in rounds (see `tools/prompt.md`). Each round ends with a comment whose last line
is `Result: gain` or `Result: no gain`. Labels: `ready` waits for a round, `running` has an agent on it,
`blocked` needs the user, `done` is finished.
