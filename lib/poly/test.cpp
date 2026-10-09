// Tests for lib/poly against O(n^2) references: transforms leaf by leaf against their definition,
// products against schoolbook multiplication, the inverse against the direct recurrence. Long
// results are checked at random coefficients (each an O(n) sum).
#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <vector>

#include "lib/poly/inverse.hpp"
#include "lib/poly/transform.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 P = poly::kModulus;
constexpr int kLgMax = 20;

std::mt19937_64 rng(2026);
int failures = 0;

void expect(bool ok, const char* what, u64 a = 0, u64 b = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL: %s (%llu, %llu)\n", what, static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
}

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 add(u32 x, u32 y) { return (x + y) % P; }
u32 sub(u32 x, u32 y) { return (x + P - y) % P; }
u32 power(u32 x, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, x = mul(x, x))
        if (e & 1) r = mul(r, x);
    return r;
}

// Random coefficients; kind 1: mostly zeros, kind 2: mostly P - 1.
std::vector<u32> random_poly(std::size_t n, int kind = 0) {
    std::vector<u32> a(n);
    for (auto& x : a) {
        const u64 r = rng();
        x = kind == 1 && r % 4 ? 0 : kind == 2 && r % 4 ? P - 1 : u32(r % P);
    }
    return a;
}

std::size_t pick(std::size_t n) { return std::size_t(rng() % n); }

// Coefficient i of a b mod (x^n - 1).
u32 cyclic_coefficient(const std::vector<u32>& a, const std::vector<u32>& b, std::size_t i) {
    const std::size_t n = a.size();
    u64 s = 0;
    for (std::size_t j = 0; j < n; ++j) s = (s + u64(a[j]) * b[(i + n - j) % n]) % P;
    return u32(s);
}

// Coefficient i of a b.
u32 product_coefficient(std::span<const u32> a, std::span<const u32> b, std::size_t i) {
    u64 s = 0;
    for (std::size_t j = 0; j <= i && j < a.size(); ++j)
        if (i - j < b.size()) s = (s + u64(a[j]) * b[i - j]) % P;
    return u32(s);
}

// Leaf p of the transform by its definition: a mod (x^8 - w_p).
std::array<u32, 8> leaf(const std::vector<u32>& a, std::size_t p, const u32* roots) {
    const u32 r = roots[ntt::detail::slot(p >> 1)], w = p & 1 ? P - r : r;
    std::array<u32, 8> c{};
    u32 wk = 1;  // w^(i / 8)
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i && i % 8 == 0) wk = mul(wk, w);
        c[i % 8] = add(c[i % 8], mul(a[i], wk));
    }
    return c;
}

struct Fixture {
    poly::Arena arena{2 * poly::Transform::words(kLgMax) + 6 * poly::Arena::footprint(std::size_t(1) << kLgMax)};
    poly::Transform t{arena, kLgMax};
    std::span<u32> buffer[4] = {arena.take(1 << kLgMax), arena.take(1 << kLgMax), arena.take(1 << kLgMax),
                                arena.take(1 << kLgMax)};
    std::span<u32> scratch = arena.take(poly::inverse_scratch(std::size_t(1) << kLgMax));
    std::span<u32> roots = arena.take(ntt::detail::table_words(kLgMax));  // for the leaf definition

    Fixture() { ntt::detail::build_table(roots.data(), (std::size_t(1) << kLgMax) / 16, ntt::detail::kRoots[0]); }

    std::span<u32> load(int i, const std::vector<u32>& a) {
        std::span<u32> s = buffer[i].first(a.size());
        std::copy(a.begin(), a.end(), s.begin());
        return s;
    }
};

bool equal(std::span<const u32> a, const std::vector<u32>& b) { return std::equal(a.begin(), a.end(), b.begin(), b.end()); }

// Leaves to check: all for short transforms, else the first, last and random ones.
std::vector<std::size_t> leaves_to_check(std::size_t n) {
    std::vector<std::size_t> ps;
    if (n <= 4096) {
        for (std::size_t p = 0; p < n / 8; ++p) ps.push_back(p);
        return ps;
    }
    ps = {0, 1, 2, 3, n / 16, n / 8 - 1};
    for (int i = 0; i < 20; ++i) ps.push_back(pick(n / 8));
    return ps;
}

void test_transforms(Fixture& fx) {
    for (int lg = poly::Transform::kMinLog; lg <= kLgMax; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (int kind = 0; kind < 3; ++kind) {
            const auto a = random_poly(n, kind), b = random_poly(n, (kind + 1) % 3), c = random_poly(n);
            // forward: leaves by definition; inverse: back to a.
            auto ta = fx.load(0, a);
            fx.t.forward(ta);
            for (std::size_t p : leaves_to_check(n)) {
                const auto want = leaf(a, p, fx.roots.data());
                expect(std::equal(want.begin(), want.end(), ta.begin() + 8 * p), "forward leaf", lg, p);
            }
            fx.t.inverse(ta);
            expect(equal(ta, a), "inverse(forward(a)) = a", lg, kind);

            // Sources x^shift in: out of place, and in place at offset shift (out's other words
            // hold garbage); unaligned shifts and sizes. Output halves.
            for (int trial = 0; trial < 6; ++trial) {
                const std::size_t shift = trial == 0 ? 0 : trial == 1 || trial == 2 ? n / 2 : pick(n);
                const std::size_t size = trial == 0 ? n / 2 : trial == 1 ? n / 2 : pick(n - shift + 1);
                const bool in_place = trial % 2 == 1;
                std::vector<u32> shifted(n, 0);
                for (std::size_t i = 0; i < size; ++i) shifted[shift + i] = a[i];
                auto want = fx.load(1, shifted);
                fx.t.forward(want);
                auto out = fx.load(0, b);  // garbage
                std::span<const u32> in;
                if (in_place) {
                    std::copy_n(a.begin(), size, out.begin() + shift);
                    in = std::span<const u32>(out).subspan(shift, size);
                } else {
                    in = std::span<const u32>(fx.load(2, a)).first(size);
                }
                fx.t.forward(in, shift, out);
                expect(std::equal(out.begin(), out.end(), want.begin()), "forward of x^shift in", lg, trial);

                const poly::Half half = trial % 3 == 0 ? poly::Half::kBoth : trial % 3 == 1 ? poly::Half::kUpper : poly::Half::kLower;
                const std::size_t lo = half == poly::Half::kUpper ? n / 2 : 0, hi = half == poly::Half::kLower ? n / 2 : n;
                fx.t.inverse(out, half);
                expect(std::equal(out.begin() + lo, out.begin() + hi, shifted.begin() + lo), "inverse of a half", lg, trial);

                auto tb = fx.load(3, b);
                fx.t.forward(tb);
                out = fx.load(0, b);
                if (in_place) {
                    std::copy_n(a.begin(), size, out.begin() + shift);
                    in = std::span<const u32>(out).subspan(shift, size);
                }
                fx.t.cyclic_product(in, shift, out, tb, half);
                for (std::size_t i : {lo, hi - 1, lo + pick(hi - lo), lo + pick(hi - lo)})
                    expect(out[i] == cyclic_coefficient(shifted, b, i), "cyclic_product of x^shift in", lg, i);
            }

            // forward_upper: leaves n/8 .. n/4 - 1 of the transform of length 2n.
            if (lg < kLgMax) {
                const std::size_t shift = kind == 0 ? 0 : pick(n), size = pick(n - shift + 1);
                std::vector<u32> shifted(n, 0);
                for (std::size_t i = 0; i < size; ++i) shifted[shift + i] = a[i];
                auto out = fx.load(0, b);
                fx.t.forward_upper(std::span<const u32>(a).first(size), shift, out);
                for (std::size_t p : leaves_to_check(n)) {
                    const auto want = leaf(shifted, n / 8 + p, fx.roots.data());
                    expect(std::equal(want.begin(), want.end(), out.begin() + 8 * p), "forward_upper leaf", lg, p);
                }
            }

            // cyclic_product, multiply and multiply_add against the cyclic product.
            std::vector<std::size_t> at;
            if (n <= 2048) {
                for (std::size_t i = 0; i < n; ++i) at.push_back(i);
            } else {
                at = {0, 1, n / 2, n - 1};
                for (int i = 0; i < 16; ++i) at.push_back(pick(n));
            }
            auto pa = fx.load(0, a), tb = fx.load(1, b);
            fx.t.forward(tb);
            fx.t.cyclic_product(pa, tb);
            for (std::size_t i : at) expect(pa[i] == cyclic_coefficient(a, b, i), "cyclic_product", lg, i);

            auto ma = fx.load(0, a), tc = fx.load(2, c);
            auto acc = fx.load(3, c);
            fx.t.forward(ma);
            fx.t.forward(tc);
            fx.t.forward(acc);
            fx.t.multiply_add(acc, ma, tb);  // c + a b
            fx.t.multiply(ma, tb);           // a b
            fx.t.multiply(tc, tc);           // c^2
            fx.t.inverse(ma);
            fx.t.inverse(acc);
            fx.t.inverse(tc);
            for (std::size_t i : at) {
                const u32 ab = cyclic_coefficient(a, b, i);
                expect(ma[i] == ab, "multiply", lg, i);
                expect(acc[i] == add(c[i], ab), "multiply_add", lg, i);
                expect(tc[i] == cyclic_coefficient(c, c, i), "multiply (squared)", lg, i);
            }
        }
    }
}

// g = 1 / f mod x^n by the recurrence.
std::vector<u32> inverse_reference(const std::vector<u32>& f, std::size_t n) {
    std::vector<u32> g(n);
    const u32 inv0 = power(f[0], P - 2);
    g[0] = inv0;
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = 0;
        for (std::size_t j = 1; j <= i && j < f.size(); ++j) s = (s + u64(f[j]) * g[i - j]) % P;
        g[i] = mul(sub(0, u32(s)), inv0);
    }
    return g;
}

void check_inverse(Fixture& fx, const std::vector<u32>& f, std::size_t n) {
    std::vector<u32> g(n, 0xFFFFFFFF);
    poly::inverse(fx.t, f, g, fx.scratch);
    if (n <= 3000) {
        expect(g == inverse_reference(f, n), "inverse", n, f.size());
        return;
    }
    const std::size_t prefix = 1000;  // g mod x^m is 1 / f mod x^m
    expect(std::equal(g.begin(), g.begin() + prefix, inverse_reference(f, prefix).begin()), "inverse prefix", n);
    std::vector<std::size_t> at = {n - 1, n - 2, n / 2, n / 2 - 1};
    for (int i = 0; i < 24; ++i) at.push_back(1 + pick(n - 1));
    for (std::size_t i : at) expect(product_coefficient(f, g, i) == 0, "f g = 1 at coefficient", n, i);
    expect(product_coefficient(f, g, 0) == 1, "f g = 1 at coefficient 0", n);
}

void test_inverse(Fixture& fx) {
    for (std::size_t n = 1; n <= 64; ++n)
        for (int kind = 0; kind < 3; ++kind) {
            auto f = random_poly(n, kind);
            if (!f[0]) f[0] = 1 + pick(P - 1);
            check_inverse(fx, f, n);
        }
    // Edge cases: f = 1, f = 1 - x (g = all ones), f shorter than n, f[0] = P - 1.
    for (std::size_t n : {65, 1000, 4096, 4097, 70000}) {
        check_inverse(fx, {1}, n);
        check_inverse(fx, {1, P - 1}, n);
        auto f = random_poly(n / 3 + 1);
        f[0] = P - 1;
        check_inverse(fx, f, n);
    }
    for (int lg = 7; lg <= kLgMax; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (std::size_t m : {n - 1, n, n + 1, n / 2 + pick(n / 2) + 1}) {
            if (m > (std::size_t(1) << kLgMax)) continue;
            auto f = random_poly(m, int(pick(3)));
            if (!f[0]) f[0] = 1;
            check_inverse(fx, f, m);
        }
    }
}

}  // namespace

int main() {
    static Fixture fx;
    test_transforms(fx);
    test_inverse(fx);
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
}
