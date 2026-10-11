// Baseline for Luogu P5383: the same route as solution.cpp (remainder tree over the points
// 0..n-1, leaves of 64 points by Horner, then one product with e^(-x)) on a textbook NTT
// (iterative radix-2, bit reversal, % P), schoolbook products for short factors, and the usual
// Newton iteration for the inverse.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Poly = std::vector<u32>;

constexpr u32 kP = 998244353, kG = 3;
constexpr std::size_t kLeaf = 64;

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
    if (a.empty() || b.empty()) return {};
    const std::size_t size = std::min(limit, a.size() + b.size() - 1);
    if (std::min(a.size(), b.size()) <= 32) {
        std::vector<u64> c(size);
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < b.size() && i + j < size; ++j) c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % kP;
        return Poly(c.begin(), c.end());
    }
    std::size_t n = 1;
    while (n < a.size() + b.size() - 1) n <<= 1;
    a.resize(n), b.resize(n);
    ntt(a, false), ntt(b, false);
    for (std::size_t i = 0; i < n; ++i) a[i] = u32(u64(a[i]) * b[i] % kP);
    ntt(a, true);
    a.resize(size);
    return a;
}

// 1 / f mod x^n: g <- g (2 - f g).
Poly inverse(const Poly& f, std::size_t n) {
    Poly g = {power(f[0], kP - 2)};
    for (std::size_t k = 1; k < n; k *= 2) {
        const std::size_t m = 2 * k;
        Poly a(f.begin(), f.begin() + std::ptrdiff_t(std::min(m, f.size())));
        Poly fg = multiply(a, g, m);
        fg.resize(m);
        for (u32& x : fg) x = x ? kP - x : 0;
        fg[0] = (fg[0] + 2) % kP;
        g = multiply(g, fg, m);
    }
    g.resize(n);
    return g;
}

struct Node {
    u32 lo, hi;
    Poly product;  // prod (x - i) over [lo, hi)
    std::unique_ptr<Node> left, right;
};

std::unique_ptr<Node> build(u32 lo, u32 hi) {
    auto node = std::make_unique<Node>(Node{lo, hi, {}, nullptr, nullptr});
    if (hi - lo <= kLeaf) {
        node->product = {1};
        for (u32 i = lo; i < hi; ++i) node->product = multiply(node->product, Poly{kP - i, 1}, ~std::size_t(0));
        return node;
    }
    const u32 mid = lo + (hi - lo) / 2;
    node->left = build(lo, mid);
    node->right = build(mid, hi);
    node->product = multiply(node->left->product, node->right->product, ~std::size_t(0));
    return node;
}

// f mod t for monic t.
Poly remainder(const Poly& f, const Poly& t) {
    const std::size_t m = t.size() - 1;
    if (f.size() <= m) return f;
    const std::size_t k = f.size() - m;
    const Poly rf(f.rbegin(), f.rbegin() + std::ptrdiff_t(k)), rt(t.rbegin(), t.rend());
    Poly q = multiply(rf, inverse(rt, k), k);
    std::reverse(q.begin(), q.end());
    const Poly qt = multiply(q, Poly(t.begin(), t.begin() + std::ptrdiff_t(m)), m);
    Poly r(f.begin(), f.begin() + std::ptrdiff_t(m));
    for (std::size_t i = 0; i < m && i < qt.size(); ++i) r[i] = (r[i] + kP - qt[i]) % kP;
    return r;
}

void descend(const Node& node, const Poly& f, std::vector<u32>& values) {
    if (!node.left) {
        for (u32 x = node.lo; x < node.hi; ++x) {
            u64 y = 0;
            for (std::size_t j = f.size(); j--;) y = (y * x + f[j]) % kP;
            values[x] = u32(y);
        }
        return;
    }
    descend(*node.left, remainder(f, node.left->product), values);
    descend(*node.right, remainder(f, node.right->product), values);
}

int main() {
    u32 n;
    if (std::scanf("%u", &n) != 1) return 1;
    Poly f(n);
    for (u32& c : f) std::scanf("%u", &c);
    std::vector<u32> values(n);
    descend(*build(0, n), f, values);

    std::vector<u32> inverse_factorial(n);
    u32 factorial = 1;
    for (u32 i = 1; i < n; ++i) factorial = u32(u64(factorial) * i % kP);
    inverse_factorial[n - 1] = power(factorial, kP - 2);
    for (u32 i = n - 1; i > 0; --i) inverse_factorial[i - 1] = u32(u64(inverse_factorial[i]) * i % kP);
    Poly scaled(n), exp_minus(n);
    for (u32 i = 0; i < n; ++i) {
        scaled[i] = u32(u64(values[i]) * inverse_factorial[i] % kP);
        exp_minus[i] = i % 2 && inverse_factorial[i] ? kP - inverse_factorial[i] : inverse_factorial[i];
    }
    const Poly b = multiply(scaled, exp_minus, n);
    for (u32 i = 0; i < n; ++i) std::printf("%u%c", b[i], i + 1 < n ? ' ' : '\n');
}
