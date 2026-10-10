// Lagrange interpolation in O(N^2): f = sum_i y_i / M'(x_i) M(x) / (x - x_i), M = prod (x - x_i).
#include <cstdio>
#include <vector>

using u64 = unsigned long long;
const u64 mod = 998244353;

u64 power(u64 x, u64 e) {
    u64 r = 1;
    for (; e; e >>= 1, x = x * x % mod)
        if (e & 1) r = r * x % mod;
    return r;
}

int main() {
    int n;
    u64 a, r;
    std::scanf("%d %llu %llu", &n, &a, &r);
    std::vector<u64> x(n), y(n);
    for (auto& v : y) std::scanf("%llu", &v);
    for (int i = 0; i < n; ++i) x[i] = i ? x[i - 1] * r % mod : a;
    std::vector<u64> m(n + 1, 0), c(n, 0), q(n);
    m[0] = 1;  // m = prod (x - x_i), low degree first
    for (int i = 0; i < n; ++i)
        for (int k = i + 1; k >= 0; --k) m[k] = ((k ? m[k - 1] : 0) + mod - m[k] * x[i] % mod) % mod;
    for (int i = 0; i < n; ++i) {
        u64 derivative = 1;
        for (int j = 0; j < n; ++j)
            if (j != i) derivative = derivative * ((x[i] + mod - x[j]) % mod) % mod;
        const u64 w = y[i] * power(derivative, mod - 2) % mod;
        u64 carry = 0;  // q = m / (x - x_i) by synthetic division, from the top
        for (int k = n; k-- > 0;) q[k] = carry = (m[k + 1] + carry * x[i]) % mod;
        for (int k = 0; k < n; ++k) c[k] = (c[k] + w * q[k]) % mod;
    }
    for (int k = 0; k < n; ++k) std::printf("%llu%c", c[k], k + 1 < n ? ' ' : '\n');
    if (n == 0) std::printf("\n");
}
