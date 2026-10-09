#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n;
    std::scanf("%d", &n);
    std::vector<unsigned long long> a(n), b(n);
    for (auto& x : a) std::scanf("%llu", &x);
    unsigned long long inv = 1;  // a[0]^(mod - 2)
    for (unsigned long long base = a[0], e = mod - 2; e; e >>= 1, base = base * base % mod)
        if (e & 1) inv = inv * base % mod;
    for (int i = 0; i < n; ++i) {
        unsigned long long s = i == 0 ? 1 : 0;
        for (int j = 1; j <= i; ++j) s = (s + mod - a[j] * b[i - j] % mod) % mod;
        b[i] = s * inv % mod;
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", b[i], i + 1 < n ? ' ' : '\n');
}
