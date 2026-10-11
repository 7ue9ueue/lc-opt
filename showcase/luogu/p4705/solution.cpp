// Luogu P4705 玩游戏, n, m, t <= 10^5, 3 s. Answer k is (1/(nm)) sum_j C(k, j) S_a(j) S_b(k - j),
// with power sums S_a(j) = sum_i a_i^j. They come from S_a(j) = -j [x^j] log prod_i (1 - a_i x);
// the binomial sum is one product of exponential generating functions. prod_i (1 - a_i x) is
// built by divide and conquer, O(n log^2 n), which dominates: the standard route, and the step the
// problem's forum reports timing out with a plain NTT.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lib/easy/io.hpp"
#include "lib/easy/poly.hpp"

namespace {

using easy::Poly;

// prod_(x in values) (1 - x z).
Poly linear_product(std::span<const std::uint32_t> values) {
    if (values.size() == 1) return {1, values[0] ? easy::kMod - values[0] : 0};
    const std::size_t half = values.size() / 2;
    return easy::multiply(linear_product(values.first(half)), linear_product(values.subspan(half)));
}

// S(j) / j! for j = 0..t, where S(j) = sum_x x^j.
Poly power_sums_egf(std::span<const std::uint32_t> values, std::size_t t, const Poly& inverse_factorial) {
    Poly s = easy::log(linear_product(values), t + 1);
    s[0] = std::uint32_t(values.size());
    for (std::size_t j = 1; j <= t; ++j) s[j] = s[j] ? easy::kMod - easy::mul(std::uint32_t(j), s[j]) : 0;
    for (std::size_t j = 0; j <= t; ++j) s[j] = easy::mul(s[j], inverse_factorial[j]);
    return s;
}

std::vector<std::uint32_t> read_values(easy::Reader& in, std::size_t n) {
    std::vector<std::uint32_t> v(n);
    for (auto& x : v) x = in.read<std::uint32_t>();
    return v;
}

}  // namespace

int main() {
    easy::Reader in;
    const auto n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    const auto a = read_values(in, n), b = read_values(in, m);
    const auto t = in.read<std::uint32_t>();

    Poly factorial(t + 1), inverse_factorial(t + 1);
    factorial[0] = 1;
    for (std::uint32_t j = 1; j <= t; ++j) factorial[j] = easy::mul(factorial[j - 1], j);
    inverse_factorial[t] = easy::power(factorial[t], easy::kMod - 2);
    for (std::uint32_t j = t; j > 0; --j) inverse_factorial[j - 1] = easy::mul(inverse_factorial[j], j);

    const Poly c = easy::multiply(power_sums_egf(a, t, inverse_factorial), power_sums_egf(b, t, inverse_factorial));
    const std::uint32_t scale = easy::power(easy::mul(n, m), easy::kMod - 2);
    easy::Writer out;
    for (std::uint32_t k = 1; k <= t; ++k) out.write(easy::mul(easy::mul(c[k], factorial[k]), scale), '\n');
}
