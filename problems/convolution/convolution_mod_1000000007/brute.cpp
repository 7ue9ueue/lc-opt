#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 1000000007;
    int n, m;
    std::scanf("%d %d", &n, &m);
    std::vector<unsigned long long> a(n), b(m), c(n + m - 1);
    for (auto& x : a) std::scanf("%llu", &x);
    for (auto& x : b) std::scanf("%llu", &x);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) c[i + j] = (c[i + j] + a[i] * b[j]) % mod;
    for (int i = 0; i < n + m - 1; ++i) std::printf("%llu%c", c[i], i + 1 < n + m - 1 ? ' ' : '\n');
}
