// log f = integral of f' (1 / f): 1 / f by its recurrence, then the sparse product and the
// integral, in O(N K).
#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n, k;
    std::scanf("%d %d", &n, &k);
    std::vector<int> index(k);
    std::vector<unsigned long long> a(k);
    for (int i = 0; i < k; ++i) std::scanf("%d %llu", &index[i], &a[i]);
    std::vector<unsigned long long> h(n);  // 1 / f, with a[0] = 1
    for (int m = 0; m < n; ++m) {
        unsigned long long s = m == 0 ? 1 : 0;
        for (int j = 1; j < k && index[j] <= m; ++j) s = (s + mod - a[j] * h[m - index[j]] % mod) % mod;
        h[m] = s;
    }
    std::vector<unsigned long long> inv(n + 1, 1);
    for (int m = 2; m <= n; ++m) inv[m] = (mod - mod / m) * inv[mod % m] % mod;
    std::vector<unsigned long long> b(n, 0);
    for (int m = 1; m < n; ++m) {
        unsigned long long s = 0;  // [x^(m - 1)] f' / f
        for (int j = 1; j < k && index[j] <= m; ++j) s = (s + index[j] * a[j] % mod * h[m - index[j]]) % mod;
        b[m] = s * inv[m] % mod;
    }
    for (int m = 0; m < n; ++m) std::printf("%llu%c", b[m], m + 1 < n ? ' ' : '\n');
}
