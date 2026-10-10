// Reference: extended Euclid with long division, O(N M).
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t P = 998244353;
using Poly = std::vector<std::uint64_t>;

std::uint64_t power(std::uint64_t x, std::uint64_t e) {
    std::uint64_t r = 1;
    for (; e; e >>= 1, x = x * x % P)
        if (e & 1) r = r * x % P;
    return r;
}

void trim(Poly& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

// a -= q b with q = c x^s.
void subtract_shifted(Poly& a, const Poly& b, std::uint64_t c, std::size_t s) {
    if (a.size() < b.size() + s) a.resize(b.size() + s);
    for (std::size_t i = 0; i < b.size(); ++i) a[i + s] = (a[i + s] + P - c * b[i] % P) % P;
}

}  // namespace

int main() {
    int n, m;
    if (std::scanf("%d %d", &n, &m) != 2) return 1;
    Poly f(n), g(m);
    for (auto& x : f) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x));
    for (auto& x : g) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x));
    if (m == 1) return std::printf("0\n\n"), 0;
    trim(f);
    // Invariant: r0 = s0 f mod g, r1 = s1 f mod g.
    Poly r0 = g, s0, r1 = f, s1{1};
    while (!r1.empty()) {
        // r0 -= q r1 and s0 -= q s1, one quotient term at a time.
        const std::uint64_t lead = power(r1.back(), P - 2);
        while (r0.size() >= r1.size()) {
            const std::size_t shift = r0.size() - r1.size();
            const std::uint64_t c = r0.back() * lead % P;
            subtract_shifted(r0, r1, c, shift);
            subtract_shifted(s0, s1, c, shift);
            trim(r0);
        }
        std::swap(r0, r1);
        std::swap(s0, s1);
    }
    // r0 = gcd: invertible iff it is a constant.
    if (r0.size() != 1) return std::printf("-1\n"), 0;
    const std::uint64_t scale = power(r0[0], P - 2);
    // s0 f = r0 mod g, so h = s0 / r0 mod g; deg s0 < deg g already except when f mod g is constant.
    Poly h = s0;
    for (auto& x : h) x = x * scale % P;
    // Reduce mod g.
    const std::uint64_t glead = power(g.back(), P - 2);
    trim(h);
    while (h.size() >= g.size()) subtract_shifted(h, g, h.back() * glead % P, h.size() - g.size()), trim(h);
    std::printf("%zu\n", h.size());
    for (std::size_t i = 0; i < h.size(); ++i) std::printf("%llu%c", static_cast<unsigned long long>(h[i]), i + 1 < h.size() ? ' ' : '\n');
    if (h.empty()) std::printf("\n");
}
