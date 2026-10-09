#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n_log;
    std::scanf("%d", &n_log);
    const int n = 1 << n_log;
    std::vector<unsigned long long> a(n), b(n), c(n);
    for (auto& x : a) std::scanf("%llu", &x);
    for (auto& x : b) std::scanf("%llu", &x);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) c[i ^ j] = (c[i ^ j] + a[i] * b[j]) % mod;
    for (int i = 0; i < n; ++i) std::printf("%llu%c", c[i], i + 1 < n ? ' ' : '\n');
}
