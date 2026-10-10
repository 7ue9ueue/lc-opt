// h = 1 / f mod g with deg h < deg g, for N, M <= 50000 coefficients mod 998244353, or -1:
// lib/poly/gcd.hpp (half-gcd jumps). Output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp).
#include "lib/io/bulk32.hpp"
#include "lib/poly/gcd.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    using poly::Arena;
    Arena input(Arena::footprint(n) + Arena::footprint(m) + Arena::footprint(m) + Arena::footprint(kTextWords));
    const std::span<std::uint32_t> f = input.take(n), g = input.take(m), h = input.take(m);
    char* const text = reinterpret_cast<char*>(input.take(kTextWords).data());
    io::read_bulk(in, f.data(), n);
    io::read_bulk(in, g.data(), m);
    Arena arena(poly::inverse_mod_words(n, m));
    const std::ptrdiff_t t = poly::inverse_mod(arena, f, g, h);
    io::Writer out;
    if (t < 0) {
        out.write("-1\n");
        return;
    }
    out.write(std::uint32_t(t), '\n');
    if (t == 0) return out.write('\n');
    fields::write(out, h.data(), std::size_t(t), text);
}

}  // namespace

RUN_EARLY(solve)
