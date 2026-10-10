#include <cstdio>
#include <vector>

using u64 = unsigned long long;
const u64 mod = 998244353;

u64 power(u64 x, u64 e) {
    u64 r = 1;
    for (; e; e >>= 1, x = x * x % mod)
        if (e & 1) r = r * x % mod;
    return r;
}

// g(f) = x solved coefficient by coefficient: [x^k] g(f) = sum_(j<=k) g_j [x^k] f^j with
// [x^k] f^k = f_1^k, from the powers of f mod x^n.
int main() {
    int n;
    std::scanf("%d", &n);
    std::vector<u64> f(n);
    for (auto& x : f) std::scanf("%llu", &x);
    std::vector<std::vector<u64>> pw(n, std::vector<u64>(n, 0));  // pw[j] = f^j mod x^n
    pw[0][0] = 1;
    for (int j = 1; j < n; ++j)
        for (int a = 0; a < n; ++a)
            if (pw[j - 1][a])
                for (int b = 1; a + b < n; ++b) pw[j][a + b] = (pw[j][a + b] + pw[j - 1][a] * f[b]) % mod;
    std::vector<u64> g(n, 0);
    for (int k = 1; k < n; ++k) {
        u64 sum = k == 1;
        for (int j = 1; j < k; ++j) sum = (sum + mod - g[j] * pw[j][k] % mod) % mod;
        g[k] = sum * power(pw[k][k], mod - 2) % mod;
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", g[i], i + 1 < n ? ' ' : '\n');
}
