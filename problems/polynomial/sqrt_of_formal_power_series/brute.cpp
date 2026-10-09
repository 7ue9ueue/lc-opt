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

// A square root of a quadratic residue a != 0 (Cipolla): (t + w)^((P+1)/2) in F_P[w], w^2 = t^2 - a.
u64 square_root(u64 a) {
    u64 t = 1;
    while (power((t * t + mod - a) % mod, (mod - 1) / 2) != mod - 1) ++t;
    const u64 w2 = (t * t + mod - a) % mod;
    u64 rx = 1, ry = 0, bx = t, by = 1;  // r = rx + ry w, b = bx + by w
    for (u64 e = (mod + 1) / 2; e; e >>= 1) {
        if (e & 1) {
            const u64 x = (rx * bx + ry * by % mod * w2) % mod, y = (rx * by + ry * bx) % mod;
            rx = x, ry = y;
        }
        const u64 x = (bx * bx + by * by % mod * w2) % mod, y = 2 * bx * by % mod;
        bx = x, by = y;
    }
    return rx;
}

int main() {
    int n;
    std::scanf("%d", &n);
    std::vector<u64> a(n), b(n, 0);
    for (auto& x : a) std::scanf("%llu", &x);
    int k = 0;
    while (k < n && a[k] == 0) ++k;
    if (k < n) {
        if (k % 2 || power(a[k], (mod - 1) / 2) != 1) {
            std::printf("-1\n");
            return 0;
        }
        // s = sqrt(a / x^k) mod x^(n - k): 2 s_0 s_i = a_(k+i) - sum_(0<j<i) s_j s_(i-j)
        const int size = n - k;
        std::vector<u64> s(size);
        s[0] = square_root(a[k]);
        const u64 inv = power(2 * s[0], mod - 2);
        for (int i = 1; i < size; ++i) {
            u64 sum = a[k + i];
            for (int j = 1; j < i; ++j) sum = (sum + mod - s[j] * s[i - j] % mod) % mod;
            s[i] = sum * inv % mod;
        }
        for (int i = 0; i < size; ++i) b[k / 2 + i] = s[i];
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", b[i], i + 1 < n ? ' ' : '\n');
}
