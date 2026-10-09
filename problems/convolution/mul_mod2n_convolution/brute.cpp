#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n_log;
    std::scanf("%d", &n_log);
    const unsigned n = 1u << n_log;
    std::vector<unsigned long long> a(n), b(n), c(n);
    for (auto& x : a) std::scanf("%llu", &x);
    for (auto& x : b) std::scanf("%llu", &x);
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j) {
            const unsigned k = (i * j) & (n - 1);
            c[k] = (c[k] + a[i] * b[j]) % mod;
        }
    for (unsigned k = 0; k < n; ++k) std::printf("%llu%c", c[k], k + 1 < n ? ' ' : '\n');
}
