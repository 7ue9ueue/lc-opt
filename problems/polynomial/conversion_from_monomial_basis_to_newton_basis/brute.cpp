// Reference in O(N^2): b_k = g(p_k), then g <- g / (x - p_k), by synthetic division from g = f.
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr std::uint64_t P = 998244353;
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    std::vector<std::uint64_t> g(n), p(n);
    for (auto& v : g)
        if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&v)) != 1) return 1;
    for (auto& v : p)
        if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&v)) != 1) return 1;
    // Step k: g[k, n) holds the current quotient; g[j] += p_k g[j + 1] from the top leaves the
    // remainder in g[k] and the next quotient in g[k + 1, n).
    for (int k = 0; k < n; ++k) {
        std::uint64_t carry = 0;
        for (int j = n; j-- > k;) g[j] = carry = (g[j] + carry * p[k]) % P;
    }
    for (int k = 0; k < n; ++k) std::printf("%llu ", static_cast<unsigned long long>(g[k]));
    std::printf("\n");
}
