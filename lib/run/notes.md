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

## Log

## Sources

- The ELF `.preinit_array` section: System V ABI, "Initialization and Termination Functions"
  (gABI); glibc runs `DT_PREINIT_ARRAY` from `_dl_init` before `DT_INIT_ARRAY`.
