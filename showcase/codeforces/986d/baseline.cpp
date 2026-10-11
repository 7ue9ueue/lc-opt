// CF 986D, the same algorithm as solution.cpp on a textbook NTT: iterative radix-2, bit reversal,
// 64-bit products reduced with % P. One decimal digit per coefficient.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 P = 998244353, G = 3;

u32 power(u64 a, u64 e) {
    u64 r = 1;
    for (; e; e >>= 1, a = a * a % P)
        if (e & 1) r = r * a % P;
    return u32(r);
}

void ntt(std::vector<u32>& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const u32 w = power(G, (P - 1) / len);
        const u32 wlen = invert ? power(w, P - 2) : w;
        for (std::size_t i = 0; i < n; i += len) {
            u64 wk = 1;
            for (std::size_t k = 0; k < len / 2; ++k) {
                const u32 u = a[i + k], v = u32(a[i + k + len / 2] * wk % P);
                a[i + k] = u + v < P ? u + v : u + v - P;
                a[i + k + len / 2] = u >= v ? u - v : u + P - v;
                wk = wk * wlen % P;
            }
        }
    }
    if (invert) {
        const u64 n_inv = power(n, P - 2);
        for (auto& x : a) x = u32(x * n_inv % P);
    }
}

using Digits = std::vector<u32>;  // little-endian decimal digits, no leading zeros

void carry(Digits& c) {
    u32 carry = 0;
    for (auto& d : c) {
        carry += d;
        d = carry % 10;
        carry /= 10;
    }
    for (; carry; carry /= 10) c.push_back(carry % 10);
    while (c.size() > 1 && c.back() == 0) c.pop_back();
}

Digits square(const Digits& a) {
    const std::size_t n = 2 * a.size() - 1;
    std::size_t len = 1;
    while (len < n) len <<= 1;
    Digits c(a);
    c.resize(len);
    ntt(c, false);
    for (auto& x : c) x = u32(u64(x) * x % P);
    ntt(c, true);
    c.resize(n);
    carry(c);
    return c;
}

Digits times(Digits a, u32 k) {
    for (auto& d : a) d *= k;
    carry(a);
    return a;
}

Digits power_of_three(u64 e) {
    Digits x = {1};
    for (int bit = 63; bit >= 0; --bit) {
        if (e >> bit == 0) continue;
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

u64 solve(const std::string& s) {
    if (s.size() == 1 && s[0] <= '4') return u64(s[0] - '0');
    Digits n(s.rbegin(), s.rend());
    for (auto& d : n) d -= '0';
    u64 k = u64(std::ceil(log3(s) - 1e-6));
    Digits q = power_of_three(k - 2);
    for (; !at_least(times(q, 9), n); ++k) q = times(std::move(q), 3);
    if (at_least(times(q, 4), n)) return 3 * k - 2;
    if (at_least(times(q, 6), n)) return 3 * k - 1;
    return 3 * k;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::string s;
    std::cin >> s;
    std::printf("%llu\n", static_cast<unsigned long long>(solve(s)));
}
