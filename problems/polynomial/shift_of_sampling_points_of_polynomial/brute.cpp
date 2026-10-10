// f(c + k) in O(N^2 + N M) by Newton's forward differences: f(x) = sum_j C(x, j) (D^j f)(0).
#include <cstdio>
#include <vector>

using u64 = unsigned long long;
const u64 mod = 998244353;

int main() {
    int n, m;
    u64 c;
    if (std::scanf("%d %d %llu", &n, &m, &c) != 3) return 1;
    std::vector<u64> d(n), inv(n + 1, 1);
    for (auto& v : d)
        if (std::scanf("%llu", &v) != 1) return 1;
    for (int j = 1; j < n; ++j)  // afterwards d[j] = (D^j f)(0)
        for (int i = n - 1; i >= j; --i) d[i] = (d[i] + mod - d[i - 1]) % mod;
    for (int i = 2; i <= n; ++i) inv[i] = (mod - mod / i) * inv[mod % i] % mod;
    for (int k = 0; k < m; ++k) {
        const u64 x = (c + k) % mod;
        u64 binomial = 1, sum = 0;  // binomial = C(x, j)
        for (int j = 0; j < n; ++j) {
            sum = (sum + binomial * d[j]) % mod;
            binomial = binomial * ((x + mod - j) % mod) % mod * inv[j + 1] % mod;
        }
        std::printf("%llu%c", sum, k + 1 < m ? ' ' : '\n');
    }
}
