// 1 / f mod x^N, N <= 500000: lib/poly's Newton iteration with this problem's step (step.hpp),
// output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp). One arena holds
// every array.
#include <algorithm>

#include "lib/io/bulk32.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"
#include "problems/polynomial/inv_of_formal_power_series/step.hpp"

namespace {

// g = 1 / f mod x^k for k = 2^(lg - 1) by step::inverse, then the last step from k to n in the
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
    io::read_bulk(in, a.data(), n);
    const std::span<const std::uint32_t> f = a.first(n);
    io::Writer out;
    if (n <= k) {  // n <= 32
        step::inverse(transform, f, g.first(n), scratch);
        return fields::write(out, g.data(), n, text);
    }
    step::inverse(transform, f, g, scratch);
    step::inverse_step(transform, f, g, a.subspan(k, n - k), scratch.first(len), a);
    fields::write(out, g.data(), k, text);  // ends in a newline: the checker reads tokens
    fields::write(out, a.data() + k, n - k, text);
}

}  // namespace

RUN_EARLY(solve)
