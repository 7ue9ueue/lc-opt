// The f of degree < N <= 2^17 through N points (x_i, y_i) mod 998244353: lib/poly/interpolation.hpp.
// Output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
#include "lib/io/bulk32.hpp"
#include "lib/poly/interpolation.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    using poly::Arena;
    Arena input(Arena::footprint(2 * n) + Arena::footprint(n) + Arena::footprint(kTextWords));
    const std::span<std::uint32_t> tokens = input.take(2 * n), c = input.take(n);
    char* const text = reinterpret_cast<char*>(input.take(kTextWords).data());
    io::read_bulk(in, tokens.data(), 2 * n);
    Arena arena(poly::interpolate_words(n));
    poly::interpolate(arena, tokens.first(n), tokens.subspan(n), c);
    io::Writer out;
    fields::write(out, c.data(), n, text);
}

}  // namespace

RUN_EARLY(solve)
