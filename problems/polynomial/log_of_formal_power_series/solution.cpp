// log(f) mod x^N, N <= 500000: blocked division f'/f of lib/poly, output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp). One arena holds every array; the text is
// written into the scratch once log is done.
#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/poly/log.hpp"
#include "lib/run/early.hpp"
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

}  // namespace

RUN_EARLY(solve)
