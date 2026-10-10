# convolution_mod_large

N, M <= 2^24 coefficients mod 998244353; print the N + M - 1 coefficients of the product. 10 s,
1 GiB (the output file's tmpfs pages count). Inputs and outputs are ~331 MB at the maximum.

Best judged: ours, [409657](https://judge.yosupo.jp/submission/409657), 418 ms (`main.cpp` of
#310; no spike; max_random_00/01 416/418 and max_ans_zero_00 416 set it, fft_killer 403-405).
Before: [409616](https://judge.yosupo.jp/submission/409616), 427 ms (`main.cpp` of #247; no spike;
fft_killer_01 427 and fft_killer_07 422, the other large cases 404-415).
Earlier: [409343](https://judge.yosupo.jp/submission/409343), 430 ms (`main.cpp` of #127; no spike).
Same version: [409265](https://judge.yosupo.jp/submission/409265), 439 ms (clean 429).
Earlier: [409233](https://judge.yosupo.jp/submission/409233), 448 ms.
Before: [408888](https://judge.yosupo.jp/submission/408888), 452 ms, the QPoly exploration-014
program (`../SymPoly/work/ntt/yosupo_convolution_mod_large_opt.cpp`; the submitted source equals
that file, checked 2026-10-10). Next other user: 737 ms (403499). Judge phases of 406521 (the exploration-011
twin, 454 ms): parse 71, NTT 205, output 156 (`write()` ~120), ~12-15 outside `main`.

## Design

`lib/ntt` (transform length 2^25 at the maximum): `ntt::Product` when both factors fit half the
transform (all large tests), else `ntt::Convolution`. `lib/io` input, output by
`../convolution_mod/fields.hpp` (10-byte fixed-width fields), start from `.preinit_array` and
`_exit` as in `../convolution_mod`.

## Log
- 2026-10-09, claude: refactored the QPoly program onto `lib/ntt`, `lib/io` input and
  `../fixed_width.hpp` output. 54/54 official tests (`tools/judge.py` now runs large problems in
  tmpfs-sized batches); stress test 100 rounds. `lc-amd`, 11 rounds, slowest 3 cases: 424.6 ms vs
  426.7 for the QPoly exploration-014 program, ratio 0.994.
  Phases on fft_killer_04 (ms): parse 64.4 vs 64.9, transform 201.4 vs 202.6, output 144.3 vs
  142.7, exit 15.
- Output path, same case: formatter out of line (GCC did not inline it) 177.5 ms; inlined 166;
  constants through an opaque pointer (memory operands instead of rebuilt broadcasts) and fewer
  operations 156.9; 250 KB blocks passed to `io::Writer::write(std::string_view)`, which writes
  them directly (one `write()` per block instead of per 64 KiB) 144.3 (QPoly: 142.6).
  In memory, 2^25 values: 48.7 → 34.7 ms (QPoly asm: 32.3).
- Tried: a 256 KiB `io::Writer` as a template capacity, 147.6; it moved functions in aplusb's
  binary and CI blocked it (see `lib/io/notes.md`). The block writes are faster anyway.
- Tried and dropped: `std::_Exit(0)` after the flush (427.7 vs 427.8 ms); one table copy per use
  of each formatter constant (34.1 vs 34.7 ms in memory, ~15 more lines).
  Not submitted yet: 0.6% is below the judge's spread (452-461 ms over our recent submissions).
- 2026-10-09, claude: submitted the current `main.cpp`. [409233](https://judge.yosupo.jp/submission/409233):
  AC, 448 ms, 612.7 MiB (1/5), large cases 440-448. [409236](https://judge.yosupo.jp/submission/409236):
  AC, 453 ms (2/5), large cases 441-453. Judged runs sit 4-7% above `lc-amd` (424.6 ms) on every
  large case, as the QPoly program did (426.7 vs 452): a systematic offset, not jitter. A guess: page
  faults of the ~613 MiB footprint cost more on the judge. Not resubmitted further.
- 2026-10-09, claude (round 2). `lc-amd`, judge flags; phases from in-process `CLOCK_MONOTONIC`
  stamps on fft_killer_04 (scratch probe, not committed), 5 runs, ms:
  - Before: parse 61.5 (of which input page faults ~20 and zero-filling a and b's lower halves
    8.4; parse alone 41.6, 1.24 ns per token), tables 2.9, top radix-4 of a and b 12.8 (mostly
    zero-filling the upper halves), four quarters 178.5, inverse top 6.8, output 132 (format ~22,
    `write()` ~110). Exit: unmapping the input alone takes 12.
  - Per radix-4 level (forward a and b, inverse a): 9.1 + 4.5 at every size from 2^10 to 2^18
    vectors, 10.8 + 4.2 at 2^20. Memory is not the limit: fusing top levels cannot gain more than
    ~2 ms. The `forward` kernel issues ~14.6 vector ops per vector and level; 3.76 cycles measured
    against 3.66 at 4 ops per cycle. Tiles (levels below 2^10 vectors, leaf products) 94.4.
  - Kept: `fields.hpp` formatter and `.preinit_array` start. `tools/judge.py bench`, 11 rounds,
    slowest 3 cases: 428.3 -> 414.6 ms, ratio 0.974 (second run: 427.1 -> 414.8, 0.968).
  - Lost: input by 256 KiB `read()` chunks into an L2 buffer, parsed by `BulkParser` (no input
    mapping, no faults, no unmap; footprint 450 -> 293 MiB): 418.8 ms vs 414.8, ratio 0.985 vs
    0.968. Alone, reading the file costs 34 ms by `read()` against 42 by mmap + touch + unmap,
    but the parse then loses the overlap with its DRAM reads.
  - Lost: `MADV_POPULATE_READ` on the input mapping: 26.5 ms vs 22 for faulting it by touch.
  - Checks: 54/54 official tests, stress 200 rounds, ASan/UBSan on 6 official cases (incl.
    max_random_00) and pipe input.
  - Next: the transform is at the vector ALU bound of its kernels (~190 of ~415 ms); gains need
    fewer ops per butterfly or cheaper leaves (`lib/ntt`). Kernel time (input faults 20, zero
    fill 16, `write()` 110, unmap ~15) is ~160 ms.
- 2026-10-09, claude: submitted the merged `main.cpp` (#127). [409265](https://judge.yosupo.jp/submission/409265):
  AC, 439 ms, 612.9 MiB (1/5), against 448 for 409233. `lc-amd` predicted 414.6 vs 428.3 (-3.2%);
  judged -2.0%.
- 2026-10-09, claude (round 3): `lib/ntt` kernels against Zen 3 limits. No code change. `lc-amd`,
  scratch benchmarks (not committed), TSC ticks per vector in L1, 1 core cycle = 0.873 ticks:
  - Zen 3 facts (dependency-free asm loops): 4 vector ops per cycle for any mix of `vpmuludq`,
    `vpmulld`, `vpsrlq`, `vpaddd`, `vpminud`, `vpblendd` (49-op mixes in 12.3 cycles); loads take no
    vector pipe; 2 vector loads per cycle; a load across a 64-byte line takes 2 load slots.
  - Measured against that bound: `forward` 3.19 (84%), `inverse` 3.18; the bottom (`tile<256>`
    minus its 9 level passes) 33 (~64%). A dependency-free copy of the leaf product's op mix runs
    at 9.6; the real leaf loop does not go below 12.9.
  - Knob searches with `gen_kernels.py` timed on `lc-amd`: `forward` 105 schedules, best 3.13 vs
    3.17-3.19 (1-2%, ~1 ms in the program); `inverse` 42 schedules, best -0.4%. Not kept.
  - Bottom as three plain loops (forward h = 1 and leaf windows; leaf products; inverse h = 1),
    GCC intrinsics: same results, 61.7 ticks per vector for `tile<256>` vs 61.6. Phases 14.4 +
    13.6 + 3.9. The leaf loop scheduled by the generator (24 schedules, 2 or 4 leaves per
    iteration): best 12.9 vs 13.6. `bottom_first` per group: 16.7 vs 14.4 for the plain loop.
    Fusing leaves with the inverse: 20.2 vs 17.5. Estimated total gain ~2 ms; not pursued.
  - Op counts: a radix-4 butterfly is 49 vector ops (4 Shoup products of 7-8 ops, 8 for lazy
    reductions, 3 extra for `+ 2P` in differences), ~6.1 per vector per binary level; radix-2 and
    radix-8 come to ~6 as well. Signed residues would drop the `+ 2P` ops, but the reductions then
    land on intervals that do not fit the next Shoup input; not found to save ops.
  - Next (guesses): a smaller footprint (b's upper half as a reused scratch quarter, inverse
    twiddles from the forward table) saves ~4 ms of page zeroing on `lc-amd`; the judge's 6% offset
    may make it worth more there.
- 2026-10-10, audit (claude): submissions not logged before. Who submitted them is not recorded.
  "clean" is the score without launch spikes (`tools/spikes.py`), where it differs.
  - Before this repo (2026-09-27 to 2026-10-08, UTC), QPoly programs; sources compared with
    `../SymPoly/work/ntt/`:
    - [406499](https://judge.yosupo.jp/submission/406499) 09-27 11:31: AC 516 ms, 609.2 MiB,
      clean 511; `yosupo_convolution_mod_large.cpp`.
    - [406503](https://judge.yosupo.jp/submission/406503) 09-27 11:43: CE; it includes
      `io007_sse.hpp` and `large_core.hpp`, not bundled. Version not identified.
    - [406505](https://judge.yosupo.jp/submission/406505) 09-27 11:44: AC 548 ms, 609.6 MiB;
      `yosupo_convolution_mod_large_sse.cpp`.
    - [406506](https://judge.yosupo.jp/submission/406506) 09-27 11:46: CE. Version not identified.
    - [406507](https://judge.yosupo.jp/submission/406507) 09-27 11:49: CE. Version not identified.
    - [406508](https://judge.yosupo.jp/submission/406508) 09-27 11:55: AC 524 ms, 609.3 MiB,
      clean 518; `yosupo_convolution_mod_large_cerr.cpp`.
    - [406518](https://judge.yosupo.jp/submission/406518) 09-27 13:42: AC 458 ms, 612.9 MiB,
      clean 452; `yosupo_convolution_mod_large_io.cpp`.
    - [408717](https://judge.yosupo.jp/submission/408717) 10-07 17:34: AC 460 ms, 612.9 MiB,
      clean 457; `yosupo_convolution_mod_large_io_probe.cpp` (as 406521).
    - [408884](https://judge.yosupo.jp/submission/408884) 10-08 13:58: AC 458 ms, 613.0 MiB,
      clean 451; `yosupo_convolution_mod_large_opt.cpp` (as 408888).
    - [408886](https://judge.yosupo.jp/submission/408886) 10-08 14:08: AC 461 ms, 613.0 MiB,
      clean 456; the same file.
    - [408887](https://judge.yosupo.jp/submission/408887) 10-08 14:08: AC 452 ms, 613.0 MiB;
      the same file.
  - [409343](https://judge.yosupo.jp/submission/409343) 2026-10-10 01:47 UTC: AC 430 ms,
    612.8 MiB, no spike; `main.cpp` of #127 (2/5 with 409265). The 15 slowest cases take
    426-430 ms (409265: 422-429, plus two spikes at 438-439). New best judged: 430 ms (was 439).
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  412.06 → 406.55 ms (0.987). 54/54 official tests.
- 2026-10-10, claude (lib, issue #156 round 2): `ntt::Product` (`lib/ntt/product.hpp`, moved
  from `../convolution_mod`) instead of `ntt::Convolution` when both factors fit half the
  transform (all large tests). At lg 25 (nv = 4^11) it keeps the radix-4 top level and gains
  convolution_mod's bottom stage (two groups per asm statement, no weight array; -13% per group
  on Zen 3). The text buffer moved into the product's mapping. `judge.py bench`, `lc-bench`
  (EPYC 7B13), 15 rounds, slowest 3 cases: 413.44 -> 405.62 ms, ratio 0.9794. 54/54 official
  tests, stress 200 rounds, ASan/UBSan on 5 official cases (file and pipe input).
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).
- 2026-10-10, claude (lib, issue #156 round 2): input by `io::read_fixed` (`lib/io/fixed32.hpp`,
  from `../convolution_mod`): 9-digit and 1-digit inputs take a fixed-stride path. Per case on
  `lc-bench`, medians of 7 runs (ms): fft_killer_01 401.9 -> 390.3, all_same_00 381.1 -> 351.8,
  all_same_01 403.3 -> 389.5; max_random_00/01 and max_ans_zero_00 unchanged (~402), so they set the
  score. Details in `lib/io/notes.md`.
- 2026-10-10, claude (round 4). Scratch files: `lc-opt-explore/convolution_mod_large/`.
  - Submitted `main.cpp` of #247 (lc-bench, 11 rounds: 0.962 against 409343's source):
    [409616](https://judge.yosupo.jp/submission/409616) AC 427 ms (1/5).
  - Judge proxy: `lc-k68`, a c2d-standard-4 with Linux 6.8.0-1070-gcp instead of 7.0 (Ubuntu
    24.04, no Docker; static binaries built on `lc-amd`). Same binaries, medians of 5, ms
    (lc-k68 / lc-amd / judged 409616): max_random_00 410.7 / 394.3 / 415, small_and_large_01
    340.2 / 302.0 / 344, random_02 381.2 / 355.3 / 386. `lc-amd` misses 4-14% on the judge,
    `lc-k68` 1.1-1.3%. Exception: fft_killer_01 and _07 399 on `lc-k68`, judged 427 and 422 (the
    fft_killer cases are judged 404-427 while the 9-digit fixed path makes them 12 ms faster than
    max_random on both VMs; not explained).
  - Linux 6.8 against 7.0 (`lc-k68` / `lc-amd`): faulting 256 MiB of huge pages 19.2 / 14.0 ms;
    `write()` of 335 MB to tmpfs 123 / 114 ms; on a generated max_random input, parse 61.5 / 57.1,
    multiply 202 / 194, output 142 / 132, exit 12 / 15. A huge page that is read before it is
    written maps the huge zero page; on 6.8 the first write then splits it into 4 KiB pages (7.0
    allocates a huge page). The transform reads the factors' lower halves first, so the zero
    tails of small_and_large and random_* end up in 4 KiB pages: small_and_large mul 235 ms on
    6.8 against 195 on 7.0.
  - Kept: `write_first`, one store per 2 MiB page between the input and the end of what the first
    pass reads. lc-k68: small_and_large_01 340.2 -> 310.3, random_02 381.2 -> 369.6, max_random_00
    410.7 -> 410.2, fft_killer_01 399.4 -> 398.2 (ms). Touching the upper halves too cost +5 ms on
    full inputs (the zeroed pages leave the cache before the top pass writes them). lc-amd:
    unchanged. lc-bench `judge.py bench` (full cases only, where the code path is the same): 406.0
    vs 407.8 ms (1.0045, noise). 54/54 official tests, stress 200 rounds, ASan/UBSan 60 rounds and
    6 official cases (file and pipe input).
  - Floors on `lc-amd` (ms): mmap + one load per 64 bytes of max_random_00 31.3 (faults 16, DRAM
    15.6 on a second pass), unmapping it 10.7; huge page faults 20 GB/s; `write()` of 335 MB 114 ms
    (1.35 us per page) for blocks of 64 KiB to 1 MiB and any buffer alignment. `perf` on `lc-intel`:
    50% of `write()` in `shmem_add_to_page_cache`, two thirds of that on the `lock add` after the
    kernel zeroes the new page (init_on_alloc). Formatter alone 0.61-0.63 ns per value.
  - Lost or neutral (`lc-amd`, then `lc-k68`): the inverse top level fused with formatting, each
    quarter's text by `pwrite` at its offset: output 138-141 vs 137 ms (blocks of 512-4096
    vectors). b in three quarters (input halves plus one scratch quarter for quarters 2 and 3): RSS
    623 -> 590 MB, time +1 ms / +1.6. Dropping input pages behind the parser (`MADV_DONTNEED`
    every 1M tokens, while their page structs are cached): max_random_00 -2.4 ms on `lc-amd`,
    fft_killer_04 0, `lc-k68` -0.9 (noise). `MADV_SEQUENTIAL` on the input: exit -0.9 ms on
    `lc-amd`. Output through a shared mapping of the output file: 218 vs 114 ms (`lc-k68` 231 vs
    123); `fallocate` first 134.
  - Next: find why fft_killer is judged 13 ms above `lc-k68` (median of 10) while the other
    large cases sit 4-5 ms above. The judged score is the slowest of ~15 cases within 25 ms.
- 2026-10-10, claude (round 4, after #310):
  - Submitted `main.cpp` of #310: [409657](https://judge.yosupo.jp/submission/409657) AC 418 ms
    (2/5), no spike. Against 409616: small_and_large 341-345 -> 312-315, random_00/01/02
    192/227/386 -> 180/209/374, fft_killer median 412 -> 404 (max 427 -> 405), max_random 414/415
    -> 416/418, max_ans_zero 414 -> 416. `lc-k68` predicted max_random_00 410.2 and
    small_and_large_01 310.3.
  - `lc-k68` against the judge for 409343's source (static build, medians of 3, ms): fft_killer_01
    428.4 (judged 429), fft_killer_07 427.8 (426), max_random_00 427.1 (428), small_and_large_01
    354.8 (355). The +23 and +28 of fft_killer_01 and _07 in 409616 did not recur: judge noise.
    On `lc-k68`, a second large case running on another core adds 4-15 ms (4 pairs).
  - Next: the score is now the mixed-length parse (max_random_00/01, max_ans_zero_00: 410 on
    `lc-k68`, judged 416-418) against the fixed path (fft_killer: 399, judged 403-405). A parse as
    fast as the fixed path is worth up to ~12 ms; time it on `lc-k68`.
- 2026-10-10, claude (lib, issue #156 round 3): `write_first` moved to `lib/mem/write_first.hpp`
  (`mem::write_first`). The function is instruction-identical; `convolve` keeps two values in
  callee-saved registers across its calls (+4 instructions). `judge.py bench`, `lc-bench`:
  407.83 -> 410.25 ms (15 rounds), then 408.28 -> 406.30 with the sources swapped (21 rounds):
  noise. 54/54 official tests. Details: `lib/mem/notes.md`.
