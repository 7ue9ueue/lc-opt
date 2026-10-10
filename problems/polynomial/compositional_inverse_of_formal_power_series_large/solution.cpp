// The compositional inverse of f mod x^N for N <= 131072, f[0] = 0, f[1] != 0: Lagrange inversion
// from Kinoshita and Li's power projection (lib/poly/compositional_inverse.hpp). Input by
// lib/io/bulk32.hpp; output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
// One arena holds every array.
#include <algorithm>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/poly/compositional_inverse.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const int lg = poly::compositional_inverse_log(n);
    const std::size_t scratch = poly::compositional_inverse_scratch(n);
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    const std::size_t f_words = std::max(n, kTextWords);  // f's span holds the output text afterwards
    poly::Arena arena(poly::Transform::words(lg) + poly::Arena::footprint(f_words) + poly::Arena::footprint(n) + scratch);
    const poly::Transform transform(arena, lg);
    const std::span<std::uint32_t> text = arena.take(f_words), f = text.first(n), g = arena.take(n);
    io::read_bulk(in, f.data(), n);
    poly::compositional_inverse(transform, f, g, arena.take(scratch));
    io::Writer out;
    fields::write(out, g.data(), n, reinterpret_cast<char*>(text.data()));
}

}  // namespace

RUN_EARLY(solve)
