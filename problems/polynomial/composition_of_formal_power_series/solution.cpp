// f(g) mod x^N for N <= 8000 and g[0] = 0: Kinoshita and Li's algorithm (lib/poly/composition.hpp).
// One arena holds every array.
#include <unistd.h>

#include "lib/io/io.hpp"
#include "lib/poly/composition.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const int lg = poly::compose_log(n);
    poly::Arena arena(poly::Transform::words(lg) + 3 * poly::Arena::footprint(n) + poly::compose_scratch(n));
    const poly::Transform transform(arena, lg);
    const std::span<std::uint32_t> f = arena.take(n), g = arena.take(n), h = arena.take(n);
    in.read(f.data(), n);
    in.read(g.data(), n);
    poly::compose(transform, f, g, h, arena.take(poly::compose_scratch(n)));
    io::Writer out;
    out.write_array(h.data(), n, ' ');
    out.write('\n');
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here). _exit skips their teardown too.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
