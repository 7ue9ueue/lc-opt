# inv_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] != 0; print the first N coefficients of 1/f.
10 s. Largest tests: max_random_* (N = 500000, transforms up to 2^19).

Best judged: none yet. Record when opened (issue #62): 25 ms.

## Design

- `lib/poly/inverse.hpp`: Newton iteration, 5 transforms of length 2k per step k -> 2k
  (lib/poly/notes.md).
- `lib/io` input; output in 10-byte fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for the tables,
  f, g, the scratch and the text.
- The program runs from `.preinit_array` and ends with `_exit` (as convolution_mod).

## Floor

`lc-amd`, max_random_00, in-process phases (ms, `CLOCK_MONOTONIC` stamps, scratch probe): parse 0.84,
tables 0.07, inverse 8.26, format 0.32 (to /dev/null); total 9.5. Whole process 13.2 ms, so start,
`write()` of 5 MB and exit take ~3.7. Floor without the inverse: ~5 ms.

## Log

- 2026-10-09, claude (round 1): first solution, with lib/poly built for it (issue #95).
  - Checks: 25/25 official tests (`tools/judge.py test`); `stress.py` 400 rounds (N <= 3000
    against `brute.cpp`, every tenth round N up to 500000 checked by f g = 1 at random
    coefficients); ASan/UBSan on 8 official cases, file and pipe input; lib/poly tests.
  - `lc-amd`, `tools/judge.py bench`, slowest 3 cases (ms, medians): first version 13.55;
    one-half inputs and outputs at the top levels 13.35 (0.988, 15 rounds); top level reading its
    source directly (no copies) 13.32 (0.982 against the first, 31 rounds); plain transforms
    13.15 (0.990 against the previous, 21 rounds).
  - Last Newton step (2^19), ms: forward of g_k 0.64, product with f 1.77, second product 1.69,
    negation and first touch of g 0.15. Steps before it take as long again.
  - Not kept, measured (details in lib/poly/notes.md): b in window form, lib/ntt's asm
    `bottom_last`, other tile sizes, two-leaf interleaving. Not pursued: Harvey's 13/9 M(n)
    reciprocal and Schoenhage's 3k-length step (estimates in lib/poly/notes.md).
- Next: the transform levels run at ~4.5 cycles per vector per radix-4 level, near the ~3.6 cycle
  uop bound; the leaf product at ~23 cycles per leaf inside `cyclic_product` (bound ~11 by uop
  count). A scheduled asm bottom (as lib/ntt's generator does) is the largest item left; then
  writing g directly from the last inverse (0.15 ms) and a smaller arena (~1-2 MiB less first touch).
