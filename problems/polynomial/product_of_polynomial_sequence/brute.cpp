// Reference: multiplies the polynomials one by one, O(D^2).
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr std::uint64_t P = 998244353;
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    std::vector<std::uint64_t> f{1};
    for (int i = 0; i < n; ++i) {
        int d;
        if (std::scanf("%d", &d) != 1) return 1;
        std::vector<std::uint64_t> g(d + 1), h(f.size() + d, 0);
        for (auto& x : g)
            if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x)) != 1) return 1;
        for (std::size_t j = 0; j < f.size(); ++j)
            for (int k = 0; k <= d; ++k) h[j + k] = (h[j + k] + f[j] * g[k]) % P;
        f = h;
    }
    for (std::size_t j = 0; j < f.size(); ++j) std::printf("%llu%c", static_cast<unsigned long long>(f[j]), j + 1 < f.size() ? ' ' : '\n');
}
