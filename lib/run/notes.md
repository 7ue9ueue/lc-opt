# lib/run

Program entry for solutions. API and usage: the header of `early.hpp`. Tests: `lib/run/test.cpp`.

## Design

- `RUN_EARLY(solve)` puts a pointer to `run_early` in `.preinit_array`. The dynamic loader calls
  it after libc is set up and before `.init_array`, so libstdc++'s initializers (iostreams,
  locales) and the program's static constructors have not run. `run_early` calls `solve()` and
  ends with `_exit(0)`: exit handlers, libstdc++ teardown and the unmapping of the input are
  skipped (the kernel frees everything at exit).
- `main()` is defined too and calls `solve()`: reached only where the loader ignores
  `.preinit_array` (non-ELF builds).
- A macro, because the pointer, `run_early` and `main` must be definitions in the program itself,
  and `solve` lives in the program's anonymous namespace. It reopens that namespace.
- Requirements on `solve()`: it flushes its own output (lib/io's `Writer` does on destruction)
  and does not rely on static constructors of its own or of libstdc++ (iostreams).

## Measurements

From `problems/convolution/convolution_mod/notes.md` (round 1, `lc-amd`, EPYC 7B13, judge flags):
start (fork to `main`) 1.02 -> 0.92 ms, exit -0.05 ms. An empty C++ program 1.11 ms, an empty C
program 0.55; an empty program whose `.preinit_array` entry calls `_exit` 0.96.

## Users

16 convolution problems: bitwise_and, bitwise_xor, convolution_F_2_64, convolution_mod,
convolution_mod_1000000007, convolution_mod_2_64, convolution_mod_large, gcd, lcm, min_plus
concave_arbitrary, convex_arbitrary and convex_convex, mul_mod2n, mul_modp, multivariate_convolution
and its cyclic variant.

Not moved yet, with the same block: 17 polynomial problems (another session's lane; lib/poly #95).

## Log

2026-10-10, claude (issue #156, round 2): the block was the same code in 30 problems (comments
apart): `run_early`, the `.preinit_array` pointer, `main`.
- The judge's command builds byte-identical stripped executables before and after for all 15
  moved problems (GCC 15.2 image, `lc-amd`).
- `test.cpp` passes at -O2 and with ASan/UBSan (`lc-amd`). A version of it with a plain `main`
  fails (static constructors run first), so it checks the early start.

2026-10-10, claude (issue #156, round 3): bitwise_and_convolution (its round had been running in
round 2). Byte-identical stripped executable before and after (judge flags, `lc-amd`).

## Sources

- The ELF `.preinit_array` section: System V ABI, "Initialization and Termination Functions"
  (gABI); glibc runs `DT_PREINIT_ARRAY` from `_dl_init` before `DT_INIT_ARRAY`.
