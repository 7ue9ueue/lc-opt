// The Newton coefficients b_i of f (N <= 2^17 coefficients) on points p_0 .. p_(N-1) mod
// 998244353: lib/poly/newton.hpp. Output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp).
#include "lib/io/bulk32.hpp"
#include "lib/poly/newton.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    if (n == 0) {
        io::Writer out;
        return out.write('\n');
    }
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    using poly::Arena;
    Arena input(Arena::footprint(2 * n) + Arena::footprint(n) + Arena::footprint(kTextWords));
    const std::span<std::uint32_t> tokens = input.take(2 * n), b = input.take(n);
    char* const text = reinterpret_cast<char*>(input.take(kTextWords).data());
    io::read_bulk(in, tokens.data(), 2 * n);
    Arena arena(poly::to_newton_words(n));
    poly::to_newton(arena, tokens.first(n), tokens.subspan(n), b);
    io::Writer out;
    fields::write(out, b.data(), n, text);
}

}  // namespace

RUN_EARLY(solve)
