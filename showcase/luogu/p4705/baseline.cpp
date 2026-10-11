// Baseline for P4705: the algorithm of solution.cpp (divide-and-conquer product of 1 - a_i x,
// log by Newton inverse, one binomial convolution) on a textbook NTT: iterative radix-2,
// bit reversal, % P arithmetic.
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
    for (std::size_t len = 2; len <= n; len <<= 1) {
        u32 w_len = power(kG, (kP - 1) / len);
        if (invert) w_len = power(w_len, kP - 2);
        for (std::size_t i = 0; i < n; i += len) {
            u64 w = 1;
            for (std::size_t j = 0; j < len / 2; ++j) {
                const u32 u = a[i + j], v = u32(a[i + j + len / 2] * w % kP);
                a[i + j] = u + v < kP ? u + v : u + v - kP;
                a[i + j + len / 2] = u >= v ? u - v : u + kP - v;
                w = w * w_len % kP;
            }
        }
    }
    if (invert) {
        const u64 inv_n = power(n, kP - 2);
        for (auto& x : a) x = u32(x * inv_n % kP);
    }
}

Poly multiply(Poly a, Poly b) {
    if (a.empty() || b.empty()) return {};
    const std::size_t result = a.size() + b.size() - 1;
    std::size_t n = 1;
    while (n < result) n <<= 1;
    a.resize(n), b.resize(n);
    ntt(a, false), ntt(b, false);
    for (std::size_t i = 0; i < n; ++i) a[i] = u32(u64(a[i]) * b[i] % kP);
    ntt(a, true);
    a.resize(result);
    return a;
}

// 1 / f mod x^n by Newton: g <- g (2 - f g).
Poly inverse(const Poly& f, std::size_t n) {
    Poly g = {power(f[0], kP - 2)};
    for (std::size_t len = 1; len < n;) {
        len *= 2;
        Poly head(f.begin(), f.begin() + std::min(f.size(), len));
        Poly fg = multiply(head, g);
        fg.resize(len);
        for (auto& x : fg) x = x ? kP - x : 0;
        fg[0] = (fg[0] + 2) % kP;
        g = multiply(g, fg);
        g.resize(len);
    }
    g.resize(n);
    return g;
}

// log f mod x^n = integral of f' / f, f[0] = 1.
Poly log(Poly f, std::size_t n) {
    f.resize(n);
    Poly derivative(n);
    for (std::size_t i = 1; i < n; ++i) derivative[i - 1] = u32(u64(f[i]) * i % kP);
    Poly q = multiply(derivative, inverse(f, n));
    Poly g(n);
    for (std::size_t i = 1; i < n; ++i) g[i] = u32(u64(q[i - 1]) * power(i, kP - 2) % kP);
    return g;
}

Poly linear_product(const u32* values, std::size_t count) {
    if (count == 1) return {1, values[0] ? kP - values[0] : 0};
    const std::size_t half = count / 2;
    return multiply(linear_product(values, half), linear_product(values + half, count - half));
}

Poly power_sums_egf(const Poly& values, std::size_t t, const Poly& inverse_factorial) {
    Poly s = log(linear_product(values.data(), values.size()), t + 1);
    s[0] = u32(values.size());
    for (std::size_t j = 1; j <= t; ++j) s[j] = s[j] ? u32(kP - u64(s[j]) * j % kP) : 0;
    for (std::size_t j = 0; j <= t; ++j) s[j] = u32(u64(s[j]) * inverse_factorial[j] % kP);
    return s;
}

int main() {
    int n, m, t;
    if (std::scanf("%d %d", &n, &m) != 2) return 1;
    Poly a(n), b(m);
    for (auto& x : a) std::scanf("%u", &x);
    for (auto& x : b) std::scanf("%u", &x);
    if (std::scanf("%d", &t) != 1) return 1;
    Poly factorial(t + 1), inverse_factorial(t + 1);
    factorial[0] = 1;
    for (int j = 1; j <= t; ++j) factorial[j] = u32(u64(factorial[j - 1]) * j % kP);
    inverse_factorial[t] = power(factorial[t], kP - 2);
    for (int j = t; j > 0; --j) inverse_factorial[j - 1] = u32(u64(inverse_factorial[j]) * j % kP);
    const Poly c = multiply(power_sums_egf(a, t, inverse_factorial), power_sums_egf(b, t, inverse_factorial));
    const u64 scale = power(u64(n) * m % kP, kP - 2);
    for (int k = 1; k <= t; ++k) std::printf("%u\n", u32(u64(c[k]) * factorial[k] % kP * scale % kP));
}
