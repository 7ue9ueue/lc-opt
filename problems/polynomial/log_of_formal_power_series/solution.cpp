// log(f) mod x^N, N <= 500000: blocked division f'/f of lib/poly, output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp). One arena holds every array; the text is
// written into the scratch once log is done.
#include <unistd.h>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/poly/log.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const int lg = poly::log_log(n);
    constexpr std::size_t kPageWords = 4096 / sizeof(std::uint32_t);
    const std::size_t scratch_words = std::max(poly::log_scratch(n), fields::kTextBytes / sizeof(std::uint32_t) + kPageWords);
    poly::Arena arena(poly::Transform::words(lg) + poly::Arena::footprint(n) + poly::Arena::footprint(scratch_words));
    const poly::Transform transform(arena, lg);
    const std::span<std::uint32_t> g = arena.take(n), scratch = arena.take(scratch_words);
    io::read_bulk(in, g.data(), n);
    poly::log(transform, g, g, scratch);  // in place
    io::Writer out;
    const auto text = (reinterpret_cast<std::uintptr_t>(scratch.data()) + 4095) & ~std::uintptr_t(4095);  // a page
    fields::write(out, g.data(), n, reinterpret_cast<char*>(text));
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
