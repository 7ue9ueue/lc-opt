// Reference: long division, O(N M).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr std::uint64_t P = 998244353;
    const auto power = [](std::uint64_t x, std::uint64_t e) {
        std::uint64_t r = 1;
        for (; e; e >>= 1, x = x * x % P)
            if (e & 1) r = r * x % P;
        return r;
    };
    int n, m;
    if (std::scanf("%d %d", &n, &m) != 2) return 1;
    std::vector<std::uint64_t> f(n), g(m);
    for (auto& x : f) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x));
    for (auto& x : g) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x));
    const int d = m - 1;
    std::vector<std::uint64_t> q(n >= m ? n - d : 0);
    const std::uint64_t lead = power(g[d], P - 2);
    for (int i = int(q.size()) - 1; i >= 0; --i) {
        q[i] = f[i + d] * lead % P;
        for (int t = 0; t <= d; ++t) f[i + t] = (f[i + t] + P - q[i] * g[t] % P) % P;
    }
    int v = std::min(n, d);
    while (v > 0 && f[v - 1] == 0) --v;
    std::printf("%d %d\n", int(q.size()), v);
    for (std::size_t i = 0; i < q.size(); ++i) std::printf("%llu%c", static_cast<unsigned long long>(q[i]), i + 1 < q.size() ? ' ' : '\n');
    if (q.empty()) std::printf("\n");
    for (int i = 0; i < v; ++i) std::printf("%llu%c", static_cast<unsigned long long>(f[i]), i + 1 < v ? ' ' : '\n');
    if (v == 0) std::printf("\n");
}
