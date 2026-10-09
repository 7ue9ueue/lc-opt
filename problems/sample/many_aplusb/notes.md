# many_aplusb

T ≤ 10^6 lines of A, B ≤ 10^18; print A + B. Pure I/O: the benchmark for `lib/io`.

## Log

- 2026-10-09, claude: `lib/io` scalar reads and writes. EPYC 7B13 (`lc-amd`), judge-like runner:
  32.5 ms vs 270.6 ms for `scanf`/`printf`. Record 23 ms (402646).
  Time (ms): start 1.1, input pages 4.7, parse 5.9, format 10.4, `write()` 9.4.
  Next: batch the lines (bulk 64-bit parse, vector add, bulk 64-bit format).
- 2026-10-09, claude: decimal addition, no binary values. Both tokens right-aligned in 32 bytes as
  digits (load ending at the token, mask, saturating subtract), added bytewise, carries from two
  32-bit masks: `carry = ((generate << 1) + propagate) ^ propagate` on the byte-reversed sum.
  Leading zeros dropped with two `pshufb` stores; the newline is byte 31 of the same vector.
  1024 lines per `write_with` call keep the loop free of calls (constants stay in registers).
  `lc-amd`: 31.9 → 21.1 ms. ~57 instructions per line (`perf`, `lc-intel`).
  Time now (ms): start 1.2, input pages 4.7, `write()` 9.4, the rest ~6.
- 2026-10-09, claude: tried and dropped: streamed input (+2.9 ms), prefetching the input (+1-4%).
- Stress test: `python3 stress.py` against `brute.cpp` (runs of 9s, powers of ten, 0, 10^18).
  A planted carry bug fails in round 0.
- 2026-10-09, claude: submitted the decimal-addition `main.cpp` (CI ratio 0.66 against the previous
  one), [409091](https://judge.yosupo.jp/submission/409091): AC, 20 ms (1/5). Slowest cases:
  max_random, digit_random, all_max, 20 ms each. Previous record 23 ms (402646); now first.
- 2026-10-09, claude, round 2 (`lc-amd`, 21-31 rounds each; ratios against the round's base):
  - Output length from the input lengths and the carry mask, not from the digits: 0.982.
  - `std::_Exit(0)` after the last flush: 0.989.
  - Line loop in inline assembly (`perf` on `lc-intel`: 83 → 69 M instructions): 0.947. Then
    `vpminub(r, r - 10)` for the digit fix 0.986, `bt` + `adc` for the length 0.994, a branch to
    a cheaper store when the sum has 16 or more digits (all large cases) 0.984, lengths kept as
    length + 1 (table offset and `bt` index without adjustment) 0.987.
  - No gain: input read in 256 KiB chunks with `read()` (1.053), stores through a stack buffer
    (store-forwarding stalls, 1.04), one 64-byte load per line instead of separator masks
    (1.008, serial chain), no empty-token check (no change), unrolling by two (1.004), prefetch
    256 B-16 KiB ahead (0.999-1.015), output buffer 128 or 256 KiB (1.003-1.011).
  - Total: 21.4 → 19.4 ms against `main` (0.903). About 48% of user cycles wait on the first load
    of each new input block (memory); kernel work (input pages, `write()`, start) is ~14.6 ms.
