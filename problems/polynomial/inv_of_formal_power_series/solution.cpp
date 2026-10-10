// 1 / f mod x^N, N <= 500000: Newton iteration of lib/poly, output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp). One arena holds every array.
#include <unistd.h>

#include <algorithm>

#include "lib/io/io.hpp"
#include "lib/poly/inverse.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

// g = 1 / f mod x^k for k = 2^(lg - 1) by poly::inverse, then the last step from k to n in the
// buffer of f: g[k, n) ends up there, after f is read. The arrays fill 3 huge pages at N = 500000
// (2 + 2 + 1 MiB and 0.75 MiB of tables and text), each first touched once.
void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const int lg = poly::inverse_log(n);
    const std::size_t len = std::size_t(1) << lg, k = len / 2;
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    const std::size_t scratch_words = std::max(len, poly::inverse_scratch(k));
    using poly::Arena;
    Arena arena(poly::Transform::words(lg) + Arena::footprint(len) + Arena::footprint(scratch_words) +
                Arena::footprint(k) + Arena::footprint(kTextWords));
    const poly::Transform transform(arena, lg);
    const std::span<std::uint32_t> a = arena.take(len), scratch = arena.take(scratch_words), g = arena.take(k);
    char* const text = reinterpret_cast<char*>(arena.take(kTextWords).data());
    in.read(a.data(), n);
    const std::span<const std::uint32_t> f = a.first(n);
    io::Writer out;
    if (n <= k) {  // n <= 32
        poly::inverse(transform, f, g.first(n), scratch);
        return fields::write(out, g.data(), n, text);
    }
    poly::inverse(transform, f, g, scratch);
    poly::inverse_step(transform, f, g, a.subspan(k, n - k), scratch.first(len), a);
    fields::write(out, g.data(), k, text);  // ends in a newline: the checker reads tokens
    fields::write(out, a.data() + k, n - k, text);
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
