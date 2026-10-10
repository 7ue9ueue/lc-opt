// f(p_i) mod 998244353 for f of N <= 2^17 coefficients and M <= 2^17 points: the transposed
// product tree of lib/poly/evaluation.hpp. Output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp).
#include <unistd.h>

#include "lib/io/bulk32.hpp"
#include "lib/poly/evaluation.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    using poly::Arena;
    Arena input(Arena::footprint(n + m) + Arena::footprint(m) + Arena::footprint(kTextWords));
    const std::span<std::uint32_t> tokens = input.take(n + m), values = input.take(m);
    char* const text = reinterpret_cast<char*>(input.take(kTextWords).data());
    io::read_bulk(in, tokens.data(), n + m);
    Arena arena(poly::evaluate_words(n, m));
    poly::evaluate(arena, tokens.first(n), tokens.subspan(n), values);
    io::Writer out;
    fields::write(out, values.data(), m, text);
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
