// Program entry for solutions. RUN_EARLY(solve) runs solve() from the executable's
// pre-initializers (.preinit_array), before the C++ runtime initializes iostreams and locales
// (lib/io uses neither) and before any static constructor, then ends the process with _exit(0):
// no exit handlers, no teardown, no unmapping. Elsewhere it defines main() { solve(); }.
// Measurements: lib/run/notes.md.
//
//   namespace {
//   void solve() { ... }  // flushes its output (io::Writer does on destruction)
//   }  // namespace
//
//   RUN_EARLY(solve)      // at namespace scope, once per program
#pragma once

#include <unistd.h>

#ifdef __ELF__
#define RUN_EARLY(solve)                                                                                  \
    namespace {                                                                                           \
    void run_early(int, char**, char**) {                                                                 \
        solve();                                                                                          \
        ::_exit(0);                                                                                       \
    }                                                                                                     \
    [[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early; \
    }                                                                                                     \
    int main() { solve(); }  // reached only if the loader skips .preinit_array
#else
#define RUN_EARLY(solve) \
    int main() { solve(); }
#endif
