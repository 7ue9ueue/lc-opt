// Reference: Horner's rule at every point, O(N M).
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr std::uint64_t P = 998244353;
    int n, m;
    if (std::scanf("%d %d", &n, &m) != 2) return 1;
    std::vector<std::uint64_t> c(n), p(m);
    for (auto& x : c)
        if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x)) != 1) return 1;
    for (auto& x : p)
        if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x)) != 1) return 1;
    for (int i = 0; i < m; ++i) {
        std::uint64_t v = 0;
        for (int j = n; j-- > 0;) v = (v * p[i] + c[j]) % P;
        std::printf("%llu%c", static_cast<unsigned long long>(v), i + 1 < m ? ' ' : '\n');
    }
}
