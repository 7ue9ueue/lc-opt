// exp(f) mod x^N, N <= 500000: Newton iteration of lib/poly, output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp). One arena holds every array.
#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/poly/exp.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const int lg = poly::exp_log(n);
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    poly::Arena arena(poly::Transform::words(lg) + poly::Arena::footprint(n) + poly::exp_scratch(n) +
                      poly::Arena::footprint(kTextWords));
    const poly::Transform transform(arena, lg);
    const std::span<std::uint32_t> g = arena.take(n);
    io::read_bulk(in, g.data(), n);
    poly::exp(transform, g, g, arena.take(poly::exp_scratch(n)));  // in place: f is read first
    io::Writer out;
    fields::write(out, g.data(), n, reinterpret_cast<char*>(arena.take(kTextWords).data()));
}

}  // namespace

RUN_EARLY(solve)
