// Reference: Lagrange's formula in O(N^2). M = prod (x - x_i); for each i, M / (x - x_i) by
// synthetic division, scaled by y_i / prod_(j != i) (x_i - x_j).
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::uint64_t P = 998244353;

std::uint64_t power(std::uint64_t x, std::uint64_t e) {
    std::uint64_t r = 1;
    for (; e; e >>= 1, x = x * x % P)
        if (e & 1) r = r * x % P;
    return r;
}

}  // namespace

int main() {
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    std::vector<std::uint64_t> x(n), y(n);
    for (auto& v : x)
        if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&v)) != 1) return 1;
    for (auto& v : y)
        if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&v)) != 1) return 1;
    std::vector<std::uint64_t> m{1};  // M, lowest coefficient first
    for (int i = 0; i < n; ++i) {
        std::vector<std::uint64_t> next(m.size() + 1, 0);
        for (std::size_t j = 0; j < m.size(); ++j) {
            next[j + 1] = (next[j + 1] + m[j]) % P;
            next[j] = (next[j] + (P - x[i]) * m[j]) % P;
        }
        m = next;
    }
    std::vector<std::uint64_t> f(n, 0), q(n);
    for (int i = 0; i < n; ++i) {
        std::uint64_t carry = 0;  // q = M / (x - x_i), from the top
        for (int j = n; j-- > 0;) q[j] = carry = (m[j + 1] + carry * x[i]) % P;
        std::uint64_t d = 1;
        for (int j = 0; j < n; ++j)
            if (j != i) d = d * ((x[i] + P - x[j]) % P) % P;
        const std::uint64_t w = y[i] * power(d, P - 2) % P;
        for (int j = 0; j < n; ++j) f[j] = (f[j] + w * q[j]) % P;
    }
    for (int j = 0; j < n; ++j) std::printf("%llu%c", static_cast<unsigned long long>(f[j]), j + 1 < n ? ' ' : '\n');
}
