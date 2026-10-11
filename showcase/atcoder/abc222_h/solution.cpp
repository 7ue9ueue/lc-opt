// ABC222 H Beautiful Binary Tree, N <= 10^7, 3 s. By Lagrange inversion the answer is
// [x^(N-1)] (1 + 3x + x^2)^(2N) / N. The editorial's O(N) solution uses the coefficients'
// recurrence; this one computes the whole power series (1 + 3x + x^2)^(2N) mod x^N by
// exp(2N log f), O(N log N), the route the editorial says does not fit the time limit.
#include <cstdint>

#include "lib/easy/io.hpp"
#include "lib/easy/poly.hpp"

int main() {
    easy::Reader in;
    const auto n = in.read<std::uint32_t>();
    const easy::Poly f = {1, 3, 1};
    const easy::Poly g = easy::pow(f, 2 * std::uint64_t(n), n);
    easy::Writer out;
    out.write(easy::mul(g[n - 1], easy::power(n, easy::kMod - 2)), '\n');
}
