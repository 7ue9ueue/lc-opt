#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n, k;
    std::scanf("%d %d", &n, &k);
    std::vector<int> index(k);
    std::vector<unsigned long long> a(k);
    for (int i = 0; i < k; ++i) std::scanf("%d %llu", &index[i], &a[i]);
    std::vector<unsigned long long> inv(n + 1, 1);  // inv[m] = 1 / m
    for (int m = 2; m <= n; ++m) inv[m] = (mod - mod / m) * inv[mod % m] % mod;
    std::vector<unsigned long long> b(n);  // m b[m] = sum_j index[j] a[j] b[m - index[j]]
    b[0] = 1;
    for (int m = 1; m < n; ++m) {
        unsigned long long s = 0;
        for (int j = 0; j < k && index[j] <= m; ++j) s = (s + index[j] * a[j] % mod * b[m - index[j]]) % mod;
        b[m] = s * inv[m] % mod;
    }
    for (int m = 0; m < n; ++m) std::printf("%llu%c", b[m], m + 1 < n ? ' ' : '\n');
}
