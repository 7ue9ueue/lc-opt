# many_aplusb

T ≤ 10^6 lines of A, B ≤ 10^18; print A + B. Pure I/O: the benchmark for `lib/io`.

## Log

- 2026-10-09, claude: `lib/io` scalar reads and writes. EPYC 7B13 (`lc-amd`), judge-like runner:
  32.5 ms vs 270.6 ms for `scanf`/`printf`. Record 23 ms (402646).
  Time (ms): start 1.1, input pages 4.7, parse 5.9, format 10.4, `write()` 9.4.
  Next: batch the lines (bulk 64-bit parse, vector add, bulk 64-bit format).
