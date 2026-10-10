#include <cstdio>
#include <vector>

using u64 = unsigned long long;
const u64 mod = 998244353;

u64 power(u64 x, u64 e) {
    u64 r = 1;
    for (x %= mod; e; e >>= 1, x = x * x % mod)
        if (e & 1) r = r * x % mod;
    return r;
}

int main() {
    int n, k;
    u64 m;
    std::scanf("%d %d %llu", &n, &k, &m);
    std::vector<int> index(k);
    std::vector<u64> a(k);
    for (int i = 0; i < k; ++i) std::scanf("%d %llu", &index[i], &a[i]);
    std::vector<u64> b(n, 0);
    if (k == 0) {
        if (m == 0) b[0] = 1;
    } else if ((unsigned __int128)index[0] * m < unsigned(n)) {
        // f = a0 x^s (1 + sum c_j x^(d_j)); g = (1 + h)^m: n g[n] = sum_j (m d_j - (n - d_j)) c_j g[n - d_j].
        const u64 s = m == 0 ? 0 : index[0] * m, a0 = power(a[0], mod - 2);
        std::vector<u64> inv(n + 1, 1);  // inv[i] = 1 / i
        for (int i = 2; i <= n; ++i) inv[i] = (mod - mod / i) * inv[mod % i] % mod;
        std::vector<int> d(k);
        std::vector<u64> c(k);
        for (int j = 1; j < k; ++j) d[j] = index[j] - index[0], c[j] = a[j] * a0 % mod;
        const u64 mm = m % mod;
        std::vector<u64> g(n);
        g[0] = power(a[0], m);
        for (int i = 1; i < n; ++i) {
            u64 sum = 0;
            for (int j = 1; j < k && d[j] <= i; ++j) {
                const u64 coefficient = (mm * d[j] % mod + mod - (i - d[j]) % mod) % mod;
                sum = (sum + coefficient * c[j] % mod * g[i - d[j]]) % mod;
            }
            g[i] = sum * inv[i] % mod;
        }
        for (u64 i = s; i < u64(n); ++i) b[i] = g[i - s];
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", b[i], i + 1 < n ? ' ' : '\n');
}
