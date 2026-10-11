// Tests for lib/easy against O(n^2) references: products (and squares) against schoolbook
// multiplication; inverse, exp, log and pow against their defining identities and recurrences;
// evaluation against Horner's rule, interpolation by evaluating its result. io.hpp: compiles.
#include "lib/easy/poly.hpp"

#include <cstdio>
#include <random>
#include <vector>

#include "lib/easy/io.hpp"

namespace {

using easy::Poly;
constexpr std::uint32_t kP = easy::kMod;

std::mt19937_64 rng(12345);
int failures = 0;

void check(bool ok, const char* what, std::size_t n) {
    if (!ok && failures++ < 20) std::printf("FAIL %s n=%zu\n", what, n);
}

std::uint32_t mul(std::uint32_t a, std::uint32_t b) { return std::uint32_t(std::uint64_t(a) * b % kP); }
std::uint32_t add(std::uint32_t a, std::uint32_t b) { return a + b >= kP ? a + b - kP : a + b; }

Poly random_poly(std::size_t n) {
    Poly f(n);
    for (auto& c : f) c = std::uint32_t(rng() % kP);
    return f;
}

Poly naive(const Poly& a, const Poly& b, std::size_t n = SIZE_MAX) {
    if (a.empty() || b.empty()) return {};
    Poly c(std::min(n, a.size() + b.size() - 1));
    for (std::size_t i = 0; i < a.size() && i < c.size(); ++i)
        for (std::size_t j = 0; j < b.size() && i + j < c.size(); ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    return c;
}

Poly derivative(const Poly& f) {
    Poly d(f.size() > 1 ? f.size() - 1 : 0);
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = mul(f[i + 1], std::uint32_t(i + 1));
    return d;
}

std::uint32_t horner(const Poly& f, std::uint32_t x) {
    std::uint32_t v = 0;
    for (std::size_t i = f.size(); i-- > 0;) v = add(mul(v, x), f[i]);
    return v;
}

void test_multiply() {
    for (std::size_t n : {1, 2, 31, 32, 33, 64, 100, 1000, 3000}) {
        for (std::size_t m : {1, 5, 33, 64, 777, 2500}) {
            const Poly a = random_poly(n), b = random_poly(m);
            check(easy::multiply(a, b) == naive(a, b), "multiply", n * 10000 + m);
        }
        const Poly a = random_poly(n);
        check(easy::multiply(a, a) == naive(a, a), "square", n);
    }
    // a large product after small ones regrows the workspace; then small ones reuse it
    const Poly a = random_poly(200000), b = random_poly(70000), c = easy::multiply(a, b);
    for (int k = 0; k < 20; ++k) {
        const std::size_t i = rng() % c.size();
        std::uint32_t s = 0;
        for (std::size_t j = i >= a.size() ? i - a.size() + 1 : 0; j <= i && j < b.size(); ++j) s = add(s, mul(a[i - j], b[j]));
        check(c[i] == s, "multiply large", i);
    }
    const Poly x = random_poly(500), y = random_poly(300);
    check(easy::multiply(x, y) == naive(x, y), "multiply after large", 500);
    check(easy::multiply(Poly{}, y).empty(), "multiply empty", 0);
}

void test_series() {
    for (std::size_t n : {1, 2, 10, 63, 64, 65, 200, 1000, 4097}) {
        Poly f = random_poly(n);
        if (f[0] == 0) f[0] = 1;
        const Poly g = easy::inverse(f, n);
        Poly one(n);
        one[0] = 1;
        check(naive(f, g, n) == one, "inverse", n);

        Poly q = random_poly(n);
        q[0] = 0;
        const Poly e = easy::exp(q, n);
        // e' = q' e mod x^(n-1)
        check(derivative(e) == naive(derivative(q), e, n - 1), "exp", n);
        check(easy::log(e, n) == q, "log", n);

        Poly h = random_poly(n / 2 + 1);
        for (std::size_t z = 0; z < h.size() && z < 3; ++z) h[z] = 0;  // leading zeros
        for (std::uint64_t k : {0, 1, 2, 3}) {
            Poly r(n);
            r[0] = 1;
            for (std::uint64_t i = 0; i < k; ++i) r = naive(r, h, n);
            r.resize(n);
            check(easy::pow(h, k, n) == r, "pow", n * 10 + k);
        }
        Poly u = random_poly(n);
        u[0] = 7;
        const std::uint64_t k = 1000000007ull * 3;
        // u^(2k) = (u^k)^2
        const Poly uk = easy::pow(u, k, n);
        check(easy::pow(u, 2 * k, n) == naive(uk, uk, n), "pow big", n);
    }
}

void test_points() {
    for (std::size_t n : {1, 7, 100, 3000}) {
        for (std::size_t m : {1, 9, 100, 2500}) {
            const Poly f = random_poly(n), x = random_poly(m);
            const auto v = easy::evaluate(f, x);
            bool ok = true;
            for (std::size_t i = 0; i < m; ++i) ok &= v[i] == horner(f, x[i]);
            check(ok, "evaluate", n * 10000 + m);
        }
        Poly x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = std::uint32_t(i * 7919 % kP);  // distinct
        const Poly y = random_poly(n), c = easy::interpolate(x, y);
        check(easy::evaluate(c, x) == y, "interpolate", n);
    }
}

}  // namespace

int main() {
    test_multiply();
    test_series();
    test_points();
    std::printf(failures ? "%d failures\n" : "lib/easy: all tests passed\n", failures);
    return failures != 0;
}
