#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n;
    unsigned long long m;
    std::scanf("%d %llu", &n, &m);
    std::vector<unsigned long long> a(n), b(n, 0), inv(n + 1, 1);
    for (auto& x : a) std::scanf("%llu", &x);
    for (int i = 2; i <= n; ++i) inv[i] = (mod - mod / i) * inv[mod % i] % mod;
    int k = 0;
    while (k < n && a[k] == 0) ++k;
    if (m == 0) {
        b[0] = 1;
    } else if (k < n && (k == 0 || m <= (unsigned long long)(n - 1) / k)) {
        // h = a / x^k, g = h^M: h g' = M h' g, so h_0 i g_i = sum_{j=1}^{i} (M j - (i - j)) h_j g_{i-j}
        const int size = n - int(k * m);
        const unsigned long long mm = m % mod, inv0 = [&] {
            unsigned long long r = 1, x = a[k];
            for (unsigned long long e = mod - 2; e; e >>= 1, x = x * x % mod)
                if (e & 1) r = r * x % mod;
            return r;
        }();
        std::vector<unsigned long long> g(size);
        g[0] = 1;
        for (unsigned long long e = m % (mod - 1), x = a[k]; e; e >>= 1, x = x * x % mod)
            if (e & 1) g[0] = g[0] * x % mod;
        for (int i = 1; i < size; ++i) {
            unsigned long long s = 0;
            for (int j = 1; j <= i && k + j < n; ++j) {
                const unsigned long long c = (mm * j % mod + mod - (i - j)) % mod;
                s = (s + c * a[k + j] % mod * g[i - j]) % mod;
            }
            g[i] = s * inv[i] % mod * inv0 % mod;
        }
        for (int i = 0; i < size; ++i) b[k * m + i] = g[i];
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", b[i], i + 1 < n ? ' ' : '\n');
}
