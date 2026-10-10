// q and r with f = q g + r, deg r < deg g, for N, M <= 500000 coefficients mod 998244353:
// lib/poly/division.hpp. Output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
#include "lib/io/bulk32.hpp"
#include "lib/poly/division.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void write_line(io::Writer& out, const std::uint32_t* values, std::size_t count, char* text) {
    if (count) return fields::write(out, values, count, text);  // ends in a newline
    out.write('\n');
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    const std::size_t k = poly::quotient_size(n, m), size = poly::remainder_size(n, m);
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    using poly::Arena;
    Arena input(Arena::footprint(n) + Arena::footprint(m) + Arena::footprint(k) + Arena::footprint(size) +
                Arena::footprint(kTextWords));
    const std::span<std::uint32_t> f = input.take(n), g = input.take(m), q = input.take(k), r = input.take(size);
    char* const text = reinterpret_cast<char*>(input.take(kTextWords).data());
    io::read_bulk(in, f.data(), n);
    io::read_bulk(in, g.data(), m);
    Arena arena(poly::divide_words(n, m));
    poly::divide(arena, f, g, q, r);
    std::size_t v = size;  // deg r + 1
    while (v > 0 && r[v - 1] == 0) --v;
    io::Writer out;
    out.write(k, ' ', v, '\n');
    write_line(out, q.data(), k, text);
    write_line(out, r.data(), v, text);
}

}  // namespace

RUN_EARLY(solve)
