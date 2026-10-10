# lib/mem

Zero-filled memory in transparent huge pages. API and usage: the header of `huge.hpp`.
Tests: `lib/mem/test.cpp`.

## Design

- `map_huge(bytes)`: an anonymous private mapping of the size rounded up to 2 MiB plus one more
  huge page, its first 2 MiB boundary returned, `MADV_HUGEPAGE` on the rounded size. Nothing is
  touched: pages fault in on first write, so a huge page is zeroed only when used. Never unmapped;
  the programs end with `_exit`.
- `huge<T>(count)`: `map_huge` for count values of T.
- `Arena`: one `map_huge` mapping, bump allocation in 64-byte steps, no bounds check (callers size
  it from the same counts they take).
- Why huge pages: 4 KiB pages cost 0.8 ms per 2 MiB in faults on `lc-amd`, a huge page 0.04-0.11
  ms (`problems/convolution/convolution_mod/notes.md`, round 3); floors drop 15-20% (`lib/io/notes.md`).

## Users

| Problem | Before | Now |
|---|---|---|

## Log

## Sources

- The problems' former copies (git history; their notes list their measurements).
