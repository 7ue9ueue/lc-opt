// Reference: g = exp(f) from g' = f' g, i g_i = sum_(k=1..i) k f_k g_(i-k), O(n^2).
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr std::uint64_t kP = 998244353;

std::uint64_t power(std::uint64_t a, std::uint64_t e) {
    std::uint64_t r = 1;
    for (a %= kP; e; e >>= 1, a = a * a % kP)
        if (e & 1) r = r * a % kP;
    return r;
}

int main() {
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    std::vector<std::uint64_t> f(n), g(n);
    for (auto& a : f)
        if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&a)) != 1) return 1;
    g[0] = 1;
    for (int i = 1; i < n; ++i) {
        std::uint64_t s = 0;
        for (int k = 1; k <= i; ++k) s = (s + k * f[k] % kP * g[i - k]) % kP;
        g[i] = s * power(i, kP - 2) % kP;
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", static_cast<unsigned long long>(g[i]), i + 1 < n ? ' ' : '\n');
}
