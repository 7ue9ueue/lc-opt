// Prefix sum of f in O(N^2) without Bernoulli numbers: the values g(0), ..., g(N) by Horner's
// rule, their forward differences d_j, and g(x) = sum_j d_j C(x, j) expanded into monomials.
#include <cstdio>
#include <vector>

using u64 = unsigned long long;
const u64 mod = 998244353;

int main() {
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    std::vector<u64> a(n);
    for (auto& v : a)
        if (std::scanf("%llu", &v) != 1) return 1;
    std::vector<u64> d(n + 1, 0), inv(n + 2, 1);
    for (int x = 0; x < n; ++x) {  // d[x + 1] = g(x + 1) = g(x) + f(x)
        u64 value = 0;
        for (int t = n - 1; t >= 0; --t) value = (value * x + a[t]) % mod;
        d[x + 1] = (d[x] + value) % mod;
    }
    for (int j = 1; j <= n; ++j)  // afterwards d[j] = (D^j g)(0)
        for (int i = n; i >= j; --i) d[i] = (d[i] + mod - d[i - 1]) % mod;
    for (int i = 2; i <= n + 1; ++i) inv[i] = (mod - mod / i) * inv[mod % i] % mod;
    std::vector<u64> g(n + 1, 0), binomial(n + 1, 0);  // binomial: C(x, j) as a polynomial
    binomial[0] = 1;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= j; ++i) g[i] = (g[i] + d[j] * binomial[i]) % mod;
        if (j == n) break;
        // C(x, j + 1) = C(x, j) (x - j) / (j + 1)
        for (int i = j + 1; i >= 0; --i) {
            const u64 shifted = i ? binomial[i - 1] : 0;
            binomial[i] = (shifted + (mod - j) % mod * binomial[i]) % mod * inv[j + 1] % mod;
        }
    }
    for (int i = 0; i <= n; ++i) std::printf("%llu%c", g[i], i < n ? ' ' : '\n');
}
