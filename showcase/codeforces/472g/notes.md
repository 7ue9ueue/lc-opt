# Codeforces 472G Design Tutorial: Increase the Constraints

https://codeforces.com/problemset/problem/472/G. Rating 2800, tags bitmasks, data structures, fft.

## Limits

- Binary strings a, b, 1 <= |a|, |b| <= 2*10^5; q <= 4*10^5 queries (p1, p2, len), 0-based.
- Answer: Hamming distance of a[p1, p1 + len) and b[p2, p2 + len).
- 7 s, 256 MB. Codeforces: GNU G++ 14.2 on Windows, -O2, no -march.

## Intended solution

Editorial (Round 270, https://codeforces.com/blog/entry/14028; Codeforces refused our fetches, so
this is from search-result excerpts): bits as +1/-1, so one correlation gives the distance of a
pattern to every window of b. Cut a into blocks and convolve each block with b by FFT,
O(n L log n) for L blocks; queries add whole blocks and scan the ends. Most accepted solutions
are a bitset brute force, O(q len / 64).

## Our route

`solution.cpp`, on `lib/easy/multiply.hpp`:

- Blocks of B bits of a. Per block, overlap-save over b: one `easy::multiply` per chunk of b,
  product length C = 8 * bit_ceil(B), giving C - 2B + 2 windows each.
- Distances go straight into prefix sums along diagonals d = p2 - p1 (one uint32 array of
  (n / B + 1)(m + 1) words, row k holds B zeros then the windows; the predecessor of (k, j) is
  (k - 1, j - B)). A query's whole blocks cost two reads.
- Ends (< B bits each) and short queries: XOR and popcount, a's words aligned, b's words
  funnel-shifted, 4 words per AVX2 step (nibble table + `vpsadbw`).
- B = sqrt(n m / q * 4096), clamped to [n m / 2^24, 2^17]: 20238 at max size, 9 blocks.
- O((n / B) m log B + q B / 64) time, O((n / B) m) memory (at most 64 MiB of prefix sums).

`baseline.cpp`: the same code with a textbook NTT (bit reversal, radix 2, root table, `% P`)
behind the same `multiply` signature, its own B constant (16384, B = 40476 at max size).

## Evidence that the route normally fails

Weak. We found no statement that FFT solutions time out here. Our textbook baseline passes in
180 ms; at the editorial's sizes the FFT part is small. What is slow is the plain bitset brute
force with scalar unaligned words: 5.1 s on all-full-length queries (max2), close to the 7 s
limit on a faster CPU than Codeforces'.

## Measurements

lc-amd (EPYC 7B13), gcc:14 image, `-std=c++23 -O2` (Codeforces flags), median of 7 runs,
`showcase/bench.py` (main and baseline alternated inside the bench lock), 2026-10-10.

| Case | Shape | ours | baseline |
|---|---|---|---|
| max0 | random p1, p2, len | 83 ms | 156 ms |
| max1 | long queries | 95 ms | 182 ms |
| max2 | every query (0, 0, 2*10^5) | 73 ms | 170 ms |
| max3 | one block plus two ends of B - 1 bits | 89 ms | 181 ms |

References (`lc-opt-explore/showcase-472g/bench.sh`, same flags, 7 rounds, max0..max2):

| Program | max0 | max1 | max2 |
|---|---|---|---|
| I/O floor (easy::Reader, easy::Writer) | 24 | 24 | 17 |
| bitset brute force, scalar unaligned words (`bitset.cpp`) | 771 | 1411 | 5105 |
| brute force with our AVX2 end kernel, no blocks | 91 | 143 | 495 |
| ours, B = 10119 | 91 | 108 | 92 |
| baseline, B = 10119 | 404 | 428 | 384 |

B scans (kCost; ms for max0 / max1 / max2):

- ours: 2048 (B 14311) 86/101/72; 4096 (B 20238) 85/99/74; 8192 (B 28622) 85/104/85;
  16384 (B 40476) 86/112/138. Chunk factor 4, 8, 16 at B = 10119: within noise.
- baseline: 4096 (B 20238) 246/254/225; 16384 (B 40476) 160/185/172; 65536 (B 80952) 172/210/176.

Phase split of ours at B = 20238 (instrumented, earlier build): read 16 ms, blocks 22 ms,
queries (parse, prefix reads, ends, output) 70-85 ms on max0/max1.

Windows: the mingw build under Wine gives the same output (CRLF line ends) on max0..max3 in
0.10-0.12 s.

## Checks

- Samples, 500 stress cases against brute.cpp, mingw build under Wine: OK (`showcase/check.py`).
- `-fsanitize=address,undefined`, 100 small cases against brute.cpp: OK.
- main.cpp: 44122 bytes (Codeforces limit 64 KB).

## Log

- 2026-10-10, v1: one length-2^18 cyclic product per block (lib/poly Transform), queries offline
  sorted by diagonal and swept per block. 2.3 s at B = 633: the sweep is q * n / B steps.
- v2: rows of uint16 distances, queries online: 150-250 ms; scalar ends 3 ns/word because
  `std::popcount` from <bit> (above the target pragma) compiled to a libgcc call.
- v3-v5: overlap-save (lib/poly, cached chunk transforms), diagonal prefix sums, AVX2 ends.
- v6 (kept): moved to lib/easy/multiply.hpp for the 64 KB limit; no cached transforms there,
  so each chunk costs three transforms. Funnel-shifted b in place of 64 shifted copies.
- Sorting queries by diagonal did not speed up the prefix reads (10.2 ms sorted vs 9.7 ms
  unsorted, max0), so queries stay online.
- Exploration files: `~/Documents/cpp_hpc/lc-opt-explore/showcase-472g/` (variants v1-v5,
  bench.sh, phases.sh, instrument.py, results bench1.txt, bench2.txt, final*.txt).
