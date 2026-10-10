// Tests for lib/run: on ELF targets solve() runs once, before static constructors, and the
// process ends with status 0 without running exit handlers. Prints PASS; exits 1 on a failure.
#include "lib/run/early.hpp"

#include <cstdlib>
#include <cstring>

namespace {

int constructed = 0;  // zero before the static constructors run, then 1

struct MarkConstructed {
    MarkConstructed() { constructed = 1; }
} mark_constructed;

int calls = 0;

void say(const char* text) {
    if (::write(1, text, std::strlen(text)) < 0) ::_exit(2);
}

[[noreturn]] void fail(const char* text) {
    say(text);
    ::_exit(1);
}

void exit_handler() { fail("FAIL: an exit handler ran\n"); }

void solve() {
    if (++calls != 1) fail("FAIL: solve() ran twice\n");
#ifdef __ELF__
    if (constructed) fail("FAIL: static constructors ran before solve()\n");
    std::atexit(exit_handler);
#endif
    say("PASS\n");
}

}  // namespace

RUN_EARLY(solve)
