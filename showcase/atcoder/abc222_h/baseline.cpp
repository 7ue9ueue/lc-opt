// Baseline for ABC222 H: the same route as solution.cpp, (1 + 3x + x^2)^(2N) mod x^N as
// exp(2N log f), on a textbook NTT (iterative radix-2, bit reversal, % P) with the usual
// Newton iterations for inverse, log and exp.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Poly = std::vector<u32>;

constexpr u32 kP = 998244353, kG = 3;

u32 power(u64 a, u64 e) {
    u64 r = 1;
    for (a %= kP; e; e >>= 1, a = a * a % kP)
        if (e & 1) r = r * a % kP;
    return u32(r);
}

void ntt(Poly& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    std::vector<u32> w(n / 2);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const u32 root = power(invert ? power(kG, kP - 2) : kG, (kP - 1) / len);
        w[0] = 1;
        for (std::size_t k = 1; k < len / 2; ++k) w[k] = u32(u64(w[k - 1]) * root % kP);
        for (std::size_t i = 0; i < n; i += len)
            for (std::size_t k = 0; k < len / 2; ++k) {
                const u32 u = a[i + k], v = u32(u64(a[i + k + len / 2]) * w[k] % kP);
                a[i + k] = u + v >= kP ? u + v - kP : u + v;
                a[i + k + len / 2] = u >= v ? u - v : u + kP - v;
            }
    }
    if (invert) {
        const u32 inv_n = power(n, kP - 2);
        for (u32& x : a) x = u32(u64(x) * inv_n % kP);
    }
}

// a * b mod x^limit.
Poly multiply(Poly a, Poly b, std::size_t limit) {
    std::size_t n = 1;
    while (n < a.size() + b.size() - 1) n <<= 1;
    a.resize(n), b.resize(n);
    ntt(a, false), ntt(b, false);
    for (std::size_t i = 0; i < n; ++i) a[i] = u32(u64(a[i]) * b[i] % kP);
    ntt(a, true);
    a.resize(std::min(limit, n));
    return a;
}

// 1 / f mod x^n: g <- g (2 - f g).
Poly inverse(const Poly& f, std::size_t n) {
    Poly g = {power(f[0], kP - 2)};
    for (std::size_t k = 1; k < n; k *= 2) {
        const std::size_t m = 2 * k;
        Poly a(f.begin(), f.begin() + std::ptrdiff_t(std::min(m, f.size())));
        Poly fg = multiply(a, g, m);
        for (u32& x : fg) x = x ? kP - x : 0;
        fg[0] = (fg[0] + 2) % kP;
        g = multiply(g, fg, m);
    }
    g.resize(n);
    return g;
}

Poly log(const Poly& f, std::size_t n) {
    Poly d(n);
    for (std::size_t i = 1; i < std::min(n + 1, f.size()); ++i) d[i - 1] = u32(u64(f[i]) * i % kP);
    Poly q = multiply(d, inverse(f, n), n);
    Poly g(n);
    for (std::size_t i = 1; i < n; ++i) g[i] = u32(u64(q[i - 1]) * power(i, kP - 2) % kP);
    return g;
}

// exp f mod x^n: g <- g (1 - log g + f).
Poly exp(const Poly& f, std::size_t n) {
    Poly g = {1};
    for (std::size_t k = 1; k < n; k *= 2) {
        const std::size_t m = 2 * k;
        Poly h = log(g, m);
        for (std::size_t i = 0; i < m; ++i) h[i] = ((i < f.size() ? f[i] : 0) + kP - h[i]) % kP;
        h[0] = (h[0] + 1) % kP;
        g = multiply(g, h, m);
    }
    g.resize(n);
    return g;
}

int main() {
    unsigned n;
    if (std::scanf("%u", &n) != 1) return 1;
    Poly l = log(Poly{1, 3, 1}, n);
    const u32 e = u32(2 * u64(n) % kP);
    for (u32& x : l) x = u32(u64(x) * e % kP);
    const Poly g = exp(l, n);
    std::printf("%u\n", u32(u64(g[n - 1]) * power(n, kP - 2) % kP));
}
