#include <cstdio>
#include <numeric>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n;
    std::scanf("%d", &n);
    std::vector<unsigned long long> a(n + 1), b(n + 1), c(n + 1);
    for (int i = 1; i <= n; ++i) std::scanf("%llu", &a[i]);
    for (int i = 1; i <= n; ++i) std::scanf("%llu", &b[i]);
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) {
            const long long k = std::lcm<long long>(i, j);
            if (k <= n) c[k] = (c[k] + a[i] * b[j]) % mod;
        }
    for (int k = 1; k <= n; ++k) std::printf("%llu%c", c[k], k < n ? ' ' : '\n');
}
