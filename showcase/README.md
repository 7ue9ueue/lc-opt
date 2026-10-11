# Showcase

Contest problems outside Library Checker where a natural but worse NTT algorithm passes only with
our library. Per problem: `solution.cpp` on `lib/easy`, `baseline.cpp` (the same algorithm on a
textbook NTT), `brute.cpp`, `gen.py`, samples and `notes.md` (sources, evidence, raw numbers).

Criteria: the algorithm is one contestants actually write and is worse than the intended one
(preferably in complexity); on an ordinary NTT it gets TLE, on ours it passes at 50-90% of the
time limit. I/O is plain `cin`/`cout` in both programs where noted, so only the NTT differs.

## Results

Max cases on `lc-bench` (EPYC 7B13) with each judge's compiler and flags, `bench.py` medians.

| Problem | TL | Our algorithm | Intended | Ours | Baseline | Verdict |
|---|---:|---|---|---:|---:|---|
| [Luogu P7292](luogu/p7292) | 800 ms | tan, sec by D&C online convolution, O(r log^2 r) | O(r log r) | 531 ms (Luogu: ~750 ms) | 5686 ms | flip |
| [ABC222 H](atcoder/abc222_h) | 3 s | series pow, O(N log N) | O(N) | 636 ms | > 20 s | flip |
| [ABC222 H](atcoder/abc222_h_dc) | 3 s | D&C online convolution, middle products, O(N log^2 N) | O(N) | 2426 ms | 31868 ms | flip (81%) |
| [Luogu P5408](luogu/p5408) | 500 ms | D&C product of (x + i), O(n log^2 n) | O(n log n) | 41 ms | 484 ms | borderline |
| [QOJ 621](qoj/621) | 2.5 s | D&C exp, O(n log^2 n) | O(n log n) | 236 ms | 1713 ms | baseline fits |
| [CodeChef POLYEVAL](codechef/polyeval) | 3 s | remainder-tree multipoint, O(n log^2 n) | O(p log p) | 217 ms | 1551 ms | baseline fits |
| [Luogu P5383](luogu/p5383) | 2 s | remainder-tree multipoint, O(n log^2 n) | O(n log^2 n), smaller constant | 82 ms | 1082 ms | baseline fits |
| [Luogu P4705](luogu/p4705) | 3 s | D&C product, log, O(n log^2 n) | same | 44 ms | 866 ms | baseline fits |
| [CodeChef CHEFINS](codechef/chefins) | 2 s | repeated squaring, O(M log^2 M) | O(M log M) | 60 ms | 557 ms | baseline fits |
| [CF 986D](codeforces/986d) | 2 s | big integer, one digit per coefficient | same, packed digits | 64 ms | 403 ms | baseline fits |
| [CF 1770G](codeforces/1770g) | 5 s | D&C NTT, O(n log^2 n) | same | 110 ms* | 895 ms* | baseline fits |
| [CF 472G](codeforces/472g) | 7 s | sqrt blocks + NTT | same | 95 ms* | 182 ms* | baseline fits |

\* lc-amd, loaded. ABC222 H by plain D&C (no middle products) took 4.67 s: over the limit.

## Tools

- `bundle.py`: as `tools/bundle.py`, plus `<sys/mman.h>` -> `lib/easy/mman.hpp` (Windows),
  inline assembly spelled `__asm__` (Luogu compiles with `-fno-asm`), and `compact.py` for every
  judge but AtCoder (Codeforces allows 64 KB, CodeChef 50000 bytes; multiply-only bundles are
  about 40-45 KB, series ones about 180 KB).
- `check.py`: judge-like build, samples, stress against `brute.cpp`, Windows build under Wine for
  Codeforces, max-case timing.
- `bench.py`: `main.cpp` against `baseline.cpp`, alternated under the bench lock.
- `BRIEF.md`: the brief for agents writing a showcase problem.

Judges: AtCoder GCC 15.2 with `-march=native`; Codeforces GCC 14 on Windows, no `-march`
(`lib/easy/target.hpp` enables AVX2); Luogu, QOJ, CodeChef Linux GCC, assumed 13, no `-march`.
Luogu needs C++20 or later.
