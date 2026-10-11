// Baseline: the same 18 squarings as solution.cpp on a textbook NTT (iterative radix-2, bit
// reversal, % P, precomputed roots). Self-contained.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kMod = 998244353, kRoot = 3;

u32 power(u64 a, u64 e) {
    u64 r = 1;
    for (; e; e >>= 1, a = a * a % kMod)
        if (e & 1) r = r * a % kMod;
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
    std::vector<u32> w(n / 2);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const u32 step = power(kRoot, (kMod - 1) / len);
        const u32 base = invert ? power(step, kMod - 2) : step;
        w[0] = 1;
        for (std::size_t k = 1; k < len / 2; ++k) w[k] = u32(u64(w[k - 1]) * base % kMod);
        for (std::size_t i = 0; i < n; i += len)
            for (std::size_t k = 0; k < len / 2; ++k) {
                const u32 u = a[i + k], v = u32(u64(a[i + k + len / 2]) * w[k] % kMod);
                a[i + k] = u + v >= kMod ? u + v - kMod : u + v;
                a[i + k + len / 2] = u >= v ? u - v : u + kMod - v;
            }
    }
    if (invert) {
        const u64 inv_n = power(n, kMod - 2);
        for (auto& x : a) x = u32(x * inv_n % kMod);
    }
}

int main() {
    int n, k, q;
    if (std::scanf("%d %d %d", &n, &k, &q) != 3) return 1;
    std::vector<int> allowed(k), queries(q);
    for (auto& f : allowed) std::scanf("%d", &f);
    for (auto& x : queries) std::scanf("%d", &x);
    const u32 m = u32(*std::max_element(queries.begin(), queries.end()));
    const std::size_t len = std::bit_ceil(2 * std::size_t(m) + 1);
    std::vector<u32> a(len);
    a[0] = 1;
    for (const int f : allowed)
        if (u32(f) <= m) a[f] = 1;
    for (int round = std::bit_width(m); round > 0; --round) {
        ntt(a, false);
        for (auto& x : a) x = u32(u64(x) * x % kMod);
        ntt(a, true);
        for (std::size_t i = 0; i < len; ++i) a[i] = i <= m && a[i] != 0;
    }
    for (const int x : queries) std::puts(a[x] ? "Yes" : "No");
}
