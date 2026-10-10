// Reference: distinct-degree then equal-degree factorization (Cantor-Zassenhaus; the trace map for
// p = 2) with schoolbook arithmetic, on f itself: gcd(g, x^(p^d) - x) takes each irreducible factor
// of degree d once even when g is not square-free; multiplicities by repeated division.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using u64 = std::uint64_t;
using Poly = std::vector<u64>;

u64 P;
std::mt19937_64 rng(12345);

u64 power(u64 x, u64 e) {
    u64 r = 1 % P;
    for (x %= P; e; e >>= 1, x = x * x % P)
        if (e & 1) r = r * x % P;
    return r;
}

void trim(Poly& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

// a = q m + r; m nonzero.
void divide(Poly a, const Poly& m, Poly* q, Poly* r) {
    trim(a);
    const u64 inv = power(m.back(), P - 2);
    Poly quot(a.size() >= m.size() ? a.size() - m.size() + 1 : 0);
    while (a.size() >= m.size()) {
        const u64 c = a.back() * inv % P;
        const std::size_t s = a.size() - m.size();
        quot[s] = c;
        for (std::size_t i = 0; i < m.size(); ++i) a[s + i] = (a[s + i] + P - c * m[i] % P) % P;
        trim(a);
    }
    if (q) *q = quot;
    if (r) *r = a;
}

Poly remainder(const Poly& a, const Poly& m) {
    Poly r;
    divide(a, m, nullptr, &r);
    return r;
}

Poly multiply_mod(const Poly& a, const Poly& b, const Poly& m) {
    if (a.empty() || b.empty()) return {};
    Poly c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + a[i] * b[j]) % P;
    return remainder(c, m);
}

Poly power_mod(Poly a, u64 e, const Poly& m) {
    Poly r = remainder({1}, m);
    for (a = remainder(a, m); e; e >>= 1, a = multiply_mod(a, a, m))
        if (e & 1) r = multiply_mod(r, a, m);
    return r;
}

Poly monic(Poly a) {
    const u64 inv = power(a.back(), P - 2);
    for (auto& x : a) x = x * inv % P;
    return a;
}

Poly gcd(Poly a, Poly b) {
    trim(a), trim(b);
    while (!b.empty()) a = remainder(a, b), std::swap(a, b);
    return a.empty() ? a : monic(a);
}

Poly subtract(Poly a, const Poly& b) {
    if (a.size() < b.size()) a.resize(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) a[i] = (a[i] + P - b[i]) % P;
    trim(a);
    return a;
}

// Splits h (square-free, every factor of degree d) into its irreducible factors.
void equal_degree(const Poly& h, int d, std::vector<Poly>& out) {
    if (int(h.size()) - 1 == d) return out.push_back(h);
    for (;;) {
        Poly a(h.size() - 1);
        for (auto& x : a) x = rng() % P;
        trim(a);
        if (a.empty()) continue;
        Poly w;
        if (P == 2) {  // trace a + a^2 + ... + a^(2^(d-1))
            Poly t = a;
            w = a;
            for (int i = 1; i < d; ++i) t = multiply_mod(t, t, h), w = subtract(w, subtract({}, t));
        } else {  // a^((p^d - 1) / 2) = (a a^p ... a^(p^(d-1)))^((p-1)/2)
            Poly t = a, norm = remainder({1}, h);
            for (int i = 0; i < d; ++i) norm = multiply_mod(norm, t, h), t = power_mod(t, P, h);
            w = subtract(power_mod(norm, (P - 1) / 2, h), {1});
        }
        const Poly g = gcd(h, w);
        if (g.size() > 1 && g.size() < h.size()) {
            Poly q;
            divide(h, g, &q, nullptr);
            equal_degree(g, d, out);
            equal_degree(monic(q), d, out);
            return;
        }
    }
}

}  // namespace

int main() {
    int n;
    std::scanf("%d %llu", &n, reinterpret_cast<unsigned long long*>(&P));
    Poly f(n + 1);
    for (auto& a : f) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&a));
    std::vector<std::pair<int, Poly>> factors;
    Poly g = f, h = {0, 1};  // h = x^(p^d) mod g
    for (int d = 1; g.size() > 1; ++d) {
        if (2 * d > int(g.size()) - 1) {  // one irreducible factor, multiplicity 1 in g
            factors.push_back({0, g});
            break;
        }
        h = power_mod(remainder(h, g), P, g);
        const Poly part = gcd(g, subtract(h, {0, 1}));
        if (part.size() <= 1) continue;
        std::vector<Poly> found;
        equal_degree(part, d, found);
        for (const auto& q : found) {
            for (Poly quo, r;; g = quo) {
                divide(g, q, &quo, &r);
                if (!r.empty()) break;
            }
            factors.push_back({0, q});
        }
    }
    std::printf("%zu\n", factors.size());
    for (auto& [e, q] : factors) {
        Poly rest = f, r;
        for (e = 0;; ++e) {
            divide(rest, q, nullptr, &r);
            if (!r.empty()) break;
            divide(rest, q, &rest, nullptr);
        }
        std::printf("%d %zu", e, q.size() - 1);
        for (const u64 b : q) std::printf(" %llu", static_cast<unsigned long long>(b));
        std::printf("\n");
    }
}
