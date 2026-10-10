#include <cstdio>
#include <utility>
#include <vector>

using u64 = unsigned long long;
const u64 mod = 998244353;

u64 power(u64 x, u64 e) {
    u64 r = 1;
    for (x %= mod; e; e >>= 1, x = x * x % mod)
        if (e & 1) r = r * x % mod;
    return r;
}

// A square root of a quadratic residue a != 0 by Cipolla's method: (t + w)^((mod + 1) / 2) in
// F_p[w] / (w^2 - (t^2 - a)), with t^2 - a a non-residue.
u64 square_root(u64 a) {
    u64 t = 0;
    while (power((t * t + mod - a) % mod, (mod - 1) / 2) != mod - 1) ++t;
    const u64 w2 = (t * t + mod - a) % mod;
    std::pair<u64, u64> r{1, 0}, b{t, 1};
    const auto times = [&](std::pair<u64, u64> x, std::pair<u64, u64> y) {
        return std::pair<u64, u64>{(x.first * y.first + x.second * y.second % mod * w2) % mod,
                                   (x.first * y.second + x.second * y.first) % mod};
    };
    for (u64 e = (mod + 1) / 2; e; e >>= 1, b = times(b, b))
        if (e & 1) r = times(r, b);
    return r.first;
}

int main() {
    int n, k;
    if (std::scanf("%d %d", &n, &k) != 2) return 1;
    std::vector<u64> f(n, 0);
    int low = n;
    for (int i = 0; i < k; ++i) {
        int index;
        u64 a;
        if (std::scanf("%d %llu", &index, &a) != 2) return 1;
        f[index] = a;
        if (index < low) low = index;
    }
    std::vector<u64> g(n, 0);
    if (low < n) {
        if (low % 2 || power(f[low], (mod - 1) / 2) != 1) {
            std::printf("-1\n");
            return 0;
        }
        // u = f / x^low, s^2 = u: u s' = u' s / 2 gives u0 m s[m] = sum_j u[j] (3 j / 2 - m) s[m - j].
        const int size = n - low;
        std::vector<u64> u(f.begin() + low, f.end()), s(size), inv(size + 1, 1);
        for (int i = 2; i <= size; ++i) inv[i] = (mod - mod / i) * inv[mod % i] % mod;
        std::vector<int> nonzero;
        for (int j = 1; j < size; ++j)
            if (u[j]) nonzero.push_back(j);
        const u64 half = (mod + 1) / 2, inverse_u0 = power(u[0], mod - 2);
        s[0] = square_root(u[0]);
        if (mod - s[0] < s[0]) s[0] = mod - s[0];
        for (int m = 1; m < size; ++m) {
            u64 sum = 0;
            for (int j : nonzero) {
                if (j > m) break;
                const u64 coefficient = (3 * half % mod * j % mod + mod - m) % mod;
                sum = (sum + u[j] * coefficient % mod * s[m - j]) % mod;
            }
            s[m] = sum * inv[m] % mod * inverse_u0 % mod;
        }
        for (int m = 0; m < size; ++m) g[low / 2 + m] = s[m];
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", g[i], i + 1 < n ? ' ' : '\n');
}
