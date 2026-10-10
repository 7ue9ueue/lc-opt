// Reference: Cantor-Zassenhaus with schoolbook arithmetic, O(N^2 log p) per powering.
// g = gcd(f, x^p - x), then gcd(g, (x + a)^((p-1)/2) - 1) for random a until linear factors.
// Prints the roots in increasing order.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::uint64_t P = 998244353;
using Poly = std::vector<std::uint64_t>;

std::uint64_t power(std::uint64_t x, std::uint64_t e) {
    std::uint64_t r = 1;
    for (x %= P; e; e >>= 1, x = x * x % P)
        if (e & 1) r = r * x % P;
    return r;
}

void trim(Poly& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

// a mod m, m nonzero.
Poly remainder(Poly a, const Poly& m) {
    trim(a);
    const std::uint64_t inv = power(m.back(), P - 2);
    while (a.size() >= m.size()) {
        const std::uint64_t c = a.back() * inv % P;
        const std::size_t s = a.size() - m.size();
        for (std::size_t i = 0; i < m.size(); ++i) a[s + i] = (a[s + i] + P - c * m[i] % P) % P;
        trim(a);
    }
    return a;
}

Poly quotient(Poly a, const Poly& m) {
    trim(a);
    const std::uint64_t inv = power(m.back(), P - 2);
    Poly q(a.size() >= m.size() ? a.size() - m.size() + 1 : 0);
    while (a.size() >= m.size()) {
        const std::uint64_t c = a.back() * inv % P;
        const std::size_t s = a.size() - m.size();
        q[s] = c;
        for (std::size_t i = 0; i < m.size(); ++i) a[s + i] = (a[s + i] + P - c * m[i] % P) % P;
        trim(a);
    }
    return q;
}

Poly multiply_mod(const Poly& a, const Poly& b, const Poly& m) {
    if (a.empty() || b.empty()) return {};
    Poly c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + a[i] * b[j]) % P;
    return remainder(c, m);
}

// base^e mod m.
Poly power_mod(Poly base, std::uint64_t e, const Poly& m) {
    Poly r = remainder({1}, m);
    base = remainder(base, m);
    for (; e; e >>= 1, base = multiply_mod(base, base, m))
        if (e & 1) r = multiply_mod(r, base, m);
    return r;
}

Poly gcd(Poly a, Poly b) {
    trim(a), trim(b);
    while (!b.empty()) {
        Poly r = remainder(a, b);
        a = b, b = r;
    }
    return a;
}

// Roots of g, a product of distinct linear factors.
void split(const Poly& g, std::mt19937_64& rng, std::vector<std::uint64_t>& roots) {
    if (g.size() <= 1) return;
    if (g.size() == 2) {
        roots.push_back((P - g[0]) * power(g[1], P - 2) % P);
        return;
    }
    while (true) {
        Poly w = power_mod({rng() % P, 1}, (P - 1) / 2, g);
        if (w.empty()) w = {0};
        w[0] = (w[0] + P - 1) % P;
        const Poly d = gcd(g, w);
        if (d.size() > 1 && d.size() < g.size()) {
            split(d, rng, roots);
            split(quotient(g, d), rng, roots);
            return;
        }
    }
}

}  // namespace

int main() {
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    Poly f(std::size_t(n) + 1);
    for (auto& c : f) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&c));
    std::vector<std::uint64_t> roots;
    std::size_t zeros = 0;
    while (f[zeros] == 0) ++zeros;
    if (zeros) roots.push_back(0);
    f.erase(f.begin(), f.begin() + std::ptrdiff_t(zeros));
    if (f.size() > 1) {
        Poly h = power_mod({0, 1}, P, f);
        h.resize(std::max<std::size_t>(h.size(), 2));
        h[1] = (h[1] + P - 1) % P;
        std::mt19937_64 rng(1);
        split(gcd(f, h), rng, roots);
    }
    std::sort(roots.begin(), roots.end());
    std::printf("%zu\n", roots.size());
    for (std::size_t i = 0; i < roots.size(); ++i) std::printf("%llu%c", static_cast<unsigned long long>(roots[i]), i + 1 < roots.size() ? ' ' : '\n');
    if (roots.empty()) std::printf("\n");
}
