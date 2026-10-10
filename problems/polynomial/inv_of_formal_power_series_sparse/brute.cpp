#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n, k;
    std::scanf("%d %d", &n, &k);
    std::vector<int> index(k);
    std::vector<unsigned long long> a(k);
    for (int i = 0; i < k; ++i) std::scanf("%d %llu", &index[i], &a[i]);
    unsigned long long inv = 1;  // a[0]^(mod - 2)
    for (unsigned long long base = a[0], e = mod - 2; e; e >>= 1, base = base * base % mod)
        if (e & 1) inv = inv * base % mod;
    std::vector<unsigned long long> b(n);
    for (int m = 0; m < n; ++m) {
        unsigned long long s = m == 0 ? 1 : 0;
        for (int j = 1; j < k && index[j] <= m; ++j) s = (s + mod - a[j] * b[m - index[j]] % mod) % mod;
        b[m] = s * inv % mod;
    }
    for (int m = 0; m < n; ++m) std::printf("%llu%c", b[m], m + 1 < n ? ' ' : '\n');
}
