// f(g) mod x^N for N <= 131072 and g[0] = 0: Kinoshita and Li's algorithm (lib/poly/composition.hpp).
// Input by lib/io/bulk32.hpp; output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
// One arena holds every array.
#include <unistd.h>

#include <algorithm>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/poly/composition.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const int lg = poly::compose_log(n);
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    const std::size_t f_words = std::max(n, kTextWords);  // f's span holds the output text after compose
    poly::Arena arena(poly::Transform::words(lg) + poly::Arena::footprint(f_words) + 2 * poly::Arena::footprint(n) +
                      poly::compose_scratch(n));
    const poly::Transform transform(arena, lg);
    const std::span<std::uint32_t> text = arena.take(f_words), f = text.first(n), g = arena.take(n), h = arena.take(n);
    io::read_bulk(in, f.data(), n);
    io::read_bulk(in, g.data(), n);
    poly::compose(transform, f, g, h, arena.take(poly::compose_scratch(n)));
    io::Writer out;
    fields::write(out, h.data(), n, reinterpret_cast<char*>(text.data()));
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
