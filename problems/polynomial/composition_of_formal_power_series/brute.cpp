#include <cstdio>
#include <vector>

using u64 = unsigned long long;
const u64 mod = 998244353;

// f(g) mod x^n by Horner's rule: h = h g + f_i from i = n - 1 down to 0.
int main() {
    int n;
    std::scanf("%d", &n);
    std::vector<u64> f(n), g(n), h(n, 0);
    for (auto& x : f) std::scanf("%llu", &x);
    for (auto& x : g) std::scanf("%llu", &x);
    for (int i = n - 1; i >= 0; --i) {
        std::vector<u64> next(n, 0);
        for (int a = 0; a < n; ++a)
            for (int b = 0; a + b < n; ++b) next[a + b] = (next[a + b] + h[a] * g[b]) % mod;
        next[0] = (next[0] + f[i]) % mod;
        h = next;
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", h[i], i + 1 < n ? ' ' : '\n');
}
