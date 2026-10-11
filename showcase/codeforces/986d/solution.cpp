// CF 986D Perfect Encoding: n < 10^(1.5e6), 2 s, 256 MB. Minimize b_1 + ... + b_m over b_1 ... b_m >= n.
// The optimum is 3s plus at most one 2 or one 4, so with k the least exponent such that 3^k >= n, the
// answer is 3k - 2 if 4 * 3^(k-2) >= n, else 3k - 1 if 2 * 3^(k-1) >= n, else 3k. We compute 3^(k-2) in
// decimal: estimate k from log3 n in floating point, raise 3 to k - 2 by repeated squaring, then check the
// estimate with one multiply by 9. Each square is one NTT product with ONE DECIMAL DIGIT PER COEFFICIENT.
// The round's author wrote that digits must be grouped into blocks for an FFT solution to pass; blocks of
// 3 digits shorten every transform 3x. O(L log L) for L digits.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

#include "lib/easy/io.hpp"
#include "lib/easy/multiply.hpp"

namespace {

using Digits = std::vector<std::uint32_t>;  // little-endian decimal digits, no leading zeros

// Propagates carries through c (entries < 3.8e9, so carries stay below 2^32 / 10) and drops leading zeros.
void carry(Digits& c) {
    std::uint32_t carry = 0;
    for (auto& d : c) {
        carry += d;
        d = carry % 10;
        carry /= 10;
    }
    for (; carry; carry /= 10) c.push_back(carry % 10);
    while (c.size() > 1 && c.back() == 0) c.pop_back();
}

// a^2. Its coefficients before carrying are < 81 * a.size() < P while a.size() < 1.2e7, so the product
// modulo P is exact.
Digits square(const Digits& a) {
    Digits c = easy::multiply(a, a);
    carry(c);
    return c;
}

Digits times(Digits a, std::uint32_t k) {
    for (auto& d : a) d *= k;
    carry(a);
    return a;
}

Digits power_of_three(std::uint64_t e) {
    Digits x = {1};
    for (int bit = std::bit_width(e) - 1; bit >= 0; --bit) {
        x = square(x);
        if (e >> bit & 1) x = times(std::move(x), 3);
    }
    return x;
}

bool at_least(const Digits& a, const Digits& b) {
    if (a.size() != b.size()) return a.size() > b.size();
    for (std::size_t i = a.size(); i-- > 0;)
        if (a[i] != b[i]) return a[i] > b[i];
    return true;
}

// log3 n from the leading 18 digits of n.
double log3(std::string_view s) {
    const std::size_t lead = std::min<std::size_t>(s.size(), 18);
    double leading = 0;
    for (std::size_t i = 0; i < lead; ++i) leading = leading * 10 + (s[i] - '0');
    return (std::log10(leading) + double(s.size() - lead)) / std::log10(3.0);
}

std::uint64_t solve(std::string_view s) {
    if (s.size() == 1 && s[0] <= '4') return std::uint64_t(s[0] - '0');  // n <= 4: one factor b_1 = n
    Digits n(s.rbegin(), s.rend());
    for (auto& d : n) d -= '0';
    // The error of log3 is below 1e-9, so 3^(k-1) < n; k is 1 short only when log3 n is within 1e-6 above
    // an integer. n >= 5 makes k >= 2.
    std::uint64_t k = std::uint64_t(std::ceil(log3(s) - 1e-6));
    Digits q = power_of_three(k - 2);
    for (; !at_least(times(q, 9), n); ++k) q = times(std::move(q), 3);
    // 3^(k-1) < n <= 3^k, q = 3^(k-2)
    if (at_least(times(q, 4), n)) return 3 * k - 2;
    if (at_least(times(q, 6), n)) return 3 * k - 1;
    return 3 * k;
}

}  // namespace

int main() {
    easy::Reader in;
    easy::Writer out;
    out.write(solve(in.token()), '\n');
}
