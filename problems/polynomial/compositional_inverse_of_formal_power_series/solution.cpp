// The compositional inverse of f mod x^N for N <= 8000, f[0] = 0, f[1] != 0: Lagrange inversion
// from Kinoshita and Li's power projection (lib/poly/compositional_inverse.hpp). One arena holds
// every array.
#include <unistd.h>

#include "lib/io/io.hpp"
#include "lib/poly/compositional_inverse.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const int lg = poly::compositional_inverse_log(n);
    const std::size_t scratch = poly::compositional_inverse_scratch(n);
    poly::Arena arena(poly::Transform::words(lg) + 2 * poly::Arena::footprint(n) + scratch);
    const poly::Transform transform(arena, lg);
    const std::span<std::uint32_t> f = arena.take(n), g = arena.take(n);
    in.read(f.data(), n);
    poly::compositional_inverse(transform, f, g, arena.take(scratch));
    io::Writer out;
    out.write_array(g.data(), n, ' ');
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
