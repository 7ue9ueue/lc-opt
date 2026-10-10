#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n, m;
    std::scanf("%d %d", &n, &m);
    std::vector<unsigned long long> a(std::size_t(n) * m), b(std::size_t(n) * m);
    for (auto& x : a) std::scanf("%llu", &x);
    unsigned long long inv = 1;  // a[0]^(mod - 2)
    for (unsigned long long base = a[0], e = mod - 2; e; e >>= 1, base = base * base % mod)
        if (e & 1) inv = inv * base % mod;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            unsigned long long s = i == 0 && j == 0 ? 1 : 0;
            for (int p = 0; p <= i; ++p)
                for (int q = 0; q <= j; ++q)
                    if (p || q) s = (s + mod - a[p * m + q] * b[(i - p) * m + (j - q)] % mod) % mod;
            b[i * m + j] = s * inv % mod;
        }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) std::printf("%llu%c", b[i * m + j], j + 1 < m ? ' ' : '\n');
}
