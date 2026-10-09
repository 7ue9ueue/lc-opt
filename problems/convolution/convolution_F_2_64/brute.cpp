// c_k = sum_{i+j=k} a_i b_j in F_2[x] / (x^64 + x^4 + x^3 + x + 1), bit by bit. O(NM * 64).
#include <cstdint>
#include <cstdio>
#include <vector>

using u64 = std::uint64_t;

u64 multiply(u64 a, u64 b) {
    u64 r = 0;
    for (int i = 63; i >= 0; --i) {
        r = (r << 1) ^ (r >> 63 ? 0x1B : 0);
        if (b >> i & 1) r ^= a;
    }
    return r;
}

int main() {
    std::size_t n, m;
    if (std::scanf("%zu %zu", &n, &m) != 2) return 1;
    std::vector<u64> a(n), b(m), c(n + m - 1);
    for (auto& x : a) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x));
    for (auto& x : b) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j) c[i + j] ^= multiply(a[i], b[j]);
    for (std::size_t k = 0; k < c.size(); ++k)
        std::printf("%llu%c", static_cast<unsigned long long>(c[k]), k + 1 == c.size() ? '\n' : ' ');
}
