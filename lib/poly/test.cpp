// Tests for lib/poly against O(n^2) references: transforms leaf by leaf against their definition,
// products against schoolbook multiplication, the inverse, exp, log, power and sqrt against their recurrences,
// coefficient-wise operations against scalar code. Long results are checked at random
// coefficients (each an O(n) sum).
#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "lib/poly/calculus.hpp"
#include "lib/poly/exp.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/log.hpp"
#include "lib/poly/pow.hpp"
#include "lib/poly/sqrt.hpp"
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
    poly::Arena arena{2 * poly::Transform::words(kLgMax) + 22 * poly::Arena::footprint(std::size_t(1) << kLgMax)};
    poly::Transform t{arena, kLgMax};
    std::span<u32> buffer[4] = {arena.take(1 << kLgMax), arena.take(1 << kLgMax), arena.take(1 << kLgMax),
                                arena.take(1 << kLgMax)};
    std::span<u32> scratch = arena.take(poly::inverse_scratch(std::size_t(1) << kLgMax));
    std::span<u32> exp_scratch = arena.take(poly::exp_scratch(std::size_t(1) << kLgMax));
    std::span<u32> log_scratch = arena.take(poly::log_scratch((std::size_t(1) << kLgMax) + 1));
    std::span<u32> power_scratch = arena.take(poly::power_scratch(std::size_t(1) << kLgMax));
    std::span<u32> sqrt_scratch = arena.take(poly::sqrt_scratch(std::size_t(1) << kLgMax));
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

// The leaf kernels against scalar code: leaf_product on windows with any words in [0, P] (the
// kernel reads only words 1 .. 16) and canonical b, including all-zero, all-(P - 1) and all-P
// inputs; fill_windows on canonical leaves, including 0 and P - 1.
void test_leaf_kernels(Fixture& fx) {
    using poly::detail::Vec;
    const u32 inverse_r = power(u32((u64(1) << 32) % P), P - 2);  // 2^-32
    const auto word = [](int kind) {
        const u64 r = rng();
        return kind == 0 ? u32(r % (P + 1)) : kind == 1 ? 0 : kind == 2 ? P - 1 : kind == 3 ? P : u32(r % 2 ? P : r % 3);
    };
    for (int trial = 0; trial < 20000; ++trial) {
        const int kind = trial < 100 ? trial % 5 : 0;
        poly::detail::Window window;
        for (u32& x : window.word) x = word(kind);
        alignas(32) u32 b[8], out[8];
        for (u32& x : b) x = kind == 3 || kind == 4 ? (rng() % 2 ? P - 1 : 0) : std::min(word(kind), P - 1);
        poly::detail::store(out, poly::detail::leaf_product(window, b));
        for (int j = 0; j < 8; ++j) {
            u64 s = 0;
            for (int i = 0; i < 8; ++i) s = (s + u64(b[i]) * window.word[8 - i + j]) % P;
            expect(out[j] < 2 * P && out[j] % P == mul(u32(s), inverse_r), "leaf_product", trial, j);
        }
    }
    for (int trial = 0; trial < 2000; ++trial) {
        const std::size_t g = trial < 1000 ? std::size_t(trial) : pick(std::size_t(1) << (kLgMax - 5));
        alignas(32) u32 a[4][8];
        for (auto& leaf : a)
            for (u32& x : leaf) x = trial % 3 == 0 ? u32(rng() % 2 ? 0 : P - 1) : u32(rng() % P);
        const Vec f[4] = {poly::detail::load(a[0]), poly::detail::load(a[1]), poly::detail::load(a[2]), poly::detail::load(a[3])};
        poly::detail::Window window[4];
        poly::detail::fill_windows(window, f, fx.roots.data(), g);
        for (int t = 0; t < 4; ++t) {
            const u32 r = fx.roots[ntt::detail::slot(2 * g + t / 2)], w = t % 2 ? P - r : r;
            for (int j = 0; j < 8; ++j) {
                expect(window[t].word[8 + j] == a[t][j], "fill_windows: leaf", g, j);
                expect(window[t].word[j] <= P && window[t].word[j] % P == mul(w, a[t][j]), "fill_windows: w a", g, j);
            }
        }
    }
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

            // Out-of-place inverse; inverse_product out of place and in place on either operand,
            // with output halves; forward_product of x^shift in.
            ta = fx.load(0, a);
            tb = fx.load(1, b);
            fx.t.forward(ta);
            fx.t.forward(tb);
            auto out = fx.load(2, c);  // garbage
            fx.t.inverse(ta, out);
            expect(equal(out, a), "inverse out of place", lg, kind);
            const poly::Half half = kind == 0 ? poly::Half::kBoth : kind == 1 ? poly::Half::kLower : poly::Half::kUpper;
            const std::size_t lo = half == poly::Half::kUpper ? n / 2 : 0, hi = half == poly::Half::kLower ? n / 2 : n;
            out = fx.load(2, c);
            fx.t.inverse_product(ta, tb, out, half);
            for (std::size_t i : at)
                if (i >= lo && i < hi) expect(out[i] == cyclic_coefficient(a, b, i), "inverse_product", lg, i);
            auto tb_copy = fx.load(3, std::vector<u32>(tb.begin(), tb.end()));
            fx.t.inverse_product(ta, tb_copy, tb_copy, half);  // into b
            for (std::size_t i : at)
                if (i >= lo && i < hi) expect(tb_copy[i] == cyclic_coefficient(a, b, i), "inverse_product into b", lg, i);
            fx.t.inverse_product(ta, tb, ta);  // into a
            for (std::size_t i : at) expect(ta[i] == cyclic_coefficient(a, b, i), "inverse_product into a", lg, i);

            // inverse_product_sum of 1, 2, 3 pairs: a b, + b c, + c a; with output halves, and into c.
            ta = fx.load(0, a), tb = fx.load(1, b);
            tc = fx.load(2, c);
            fx.t.forward(ta);
            fx.t.forward(tb);
            fx.t.forward(tc);
            const poly::Transform::Pair pairs[3] = {{ta, tb}, {tb, tc}, {tc, ta}};
            for (std::size_t count = 1; count <= 3; ++count) {
                const bool into_c = count == 3 && kind == 2;
                out = into_c ? tc : fx.load(3, a);  // garbage
                fx.t.inverse_product_sum(std::span(pairs, count), out, half);
                for (std::size_t i : at) {
                    if (i < lo || i >= hi) continue;
                    u32 want = cyclic_coefficient(a, b, i);
                    if (count >= 2) want = add(want, cyclic_coefficient(b, c, i));
                    if (count >= 3) want = add(want, cyclic_coefficient(c, a, i));
                    expect(out[i] == want, into_c ? "inverse_product_sum into an operand" : "inverse_product_sum", lg, count);
                }
            }

            const std::size_t shift = kind == 0 ? 0 : pick(n), size = kind == 2 ? n - shift : pick(n - shift + 1);
            std::vector<u32> shifted(n, 0);
            for (std::size_t i = 0; i < size; ++i) shifted[shift + i] = a[i];
            const auto source = fx.load(0, a);
            out = fx.load(2, c);
            fx.t.forward_product(std::span<const u32>(source).first(size), shift, out, tb);
            for (std::size_t p : leaves_to_check(n))
                for (int k = 0; k < 8; ++k) expect(out[8 * p + k] < P, "forward_product canonical", lg, p);
            fx.t.inverse(out);
            for (std::size_t i : at) expect(out[i] == cyclic_coefficient(shifted, b, i), "forward_product", lg, i);
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

void test_derivative() {
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = trial < 100 ? std::size_t(trial) : pick(5000), size = trial % 3 == 0 ? n + 1 : pick(n + 2);
        const auto f = random_poly(size, trial % 3);
        std::vector<u32> want(n);
        for (std::size_t i = 0; i < n; ++i) want[i] = i + 1 < size ? mul(u32(i + 1), f[i + 1]) : 0;
        std::vector<u32> d(n + 1, 7);
        poly::derivative(f, std::span(d).first(n));
        expect(std::equal(want.begin(), want.end(), d.begin()) && d[n] == 7, "derivative", n, size);
        if (size >= n) {  // in place
            auto g = f;
            poly::derivative(g, std::span(g).first(n));
            expect(std::equal(want.begin(), want.end(), g.begin()), "derivative in place", n, size);
        }
    }
}

void test_divide_by_index() {
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = trial < 100 ? std::size_t(trial) : trial % 10 == 0 ? 100000 + pick(1000) : pick(3000);
        const std::size_t first = trial % 4 == 0 ? 1 : trial % 4 == 1 ? P - n : 1 + pick(P - n - 1);
        const auto a = random_poly(n, trial % 3);
        std::vector<u32> q(n + 1, 7);
        poly::divide_by_index(a, first, std::span(q).first(n));
        expect(q[n] == 7, "divide_by_index: past the end", n, first);
        std::vector<std::size_t> at;
        for (std::size_t i = 0; i < n; ++i)
            if (n <= 3000 || i < 100 || i + 100 >= n || pick(100) == 0) at.push_back(i);
        for (std::size_t i : at) expect(mul(q[i], u32(first + i)) == a[i], "divide_by_index", n, i);
    }
}

// g = exp(f) mod x^n by n g_n = sum_k k f_k g_(n-k).
std::vector<u32> exp_reference(const std::vector<u32>& f, std::size_t n) {
    std::vector<u32> g(n);
    g[0] = 1;
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = 0;
        for (std::size_t k = 1; k <= i && k < f.size(); ++k) s = (s + u64(k) * f[k] % P * g[i - k]) % P;
        g[i] = mul(u32(s), power(u32(i), P - 2));
    }
    return g;
}

void check_exp(Fixture& fx, const std::vector<u32>& f, std::size_t n, bool in_place = false) {
    std::vector<u32> g(n, 0xFFFFFFFF);
    if (in_place) {
        g = f;
        g.resize(n);
        poly::exp(fx.t, g, g, fx.exp_scratch);
    } else {
        poly::exp(fx.t, f, g, fx.exp_scratch);
    }
    if (n <= 3000) {
        expect(g == exp_reference(f, n), "exp", n, f.size());
        return;
    }
    const std::size_t prefix = 1000;  // g mod x^m is exp(f) mod x^m
    expect(std::equal(g.begin(), g.begin() + prefix, exp_reference(f, prefix).begin()), "exp prefix", n);
    std::vector<std::size_t> at = {n - 1, n - 2, n / 2, n / 2 - 1};
    for (int i = 0; i < 24; ++i) at.push_back(1 + pick(n - 1));
    for (std::size_t i : at) {  // i g_i = sum_k k f_k g_(i-k)
        u64 s = 0;
        for (std::size_t k = 1; k <= i && k < f.size(); ++k) s = (s + u64(k) * f[k] % P * g[i - k]) % P;
        expect(s == mul(u32(i), g[i]), "g' = f' g at coefficient", n, i);
    }
}

void test_exp(Fixture& fx) {
    for (std::size_t n = 1; n <= 160; ++n)
        for (int kind = 0; kind < 3; ++kind) {
            auto f = random_poly(n, kind);
            f[0] = 0;
            check_exp(fx, f, n, kind == 1);
        }
    // Edge cases: f = 0 (g = 1), f = x (g_i = 1 / i!), f shorter and longer than n, f[1] = P - 1.
    for (std::size_t n : {65, 1000, 4096, 4097, 70000}) {
        check_exp(fx, {0}, n);
        check_exp(fx, {0, 1}, n);
        auto f = random_poly(n / 3 + 1);
        f[0] = 0, f[1] = P - 1;
        check_exp(fx, f, n);
        f = random_poly(2 * n);
        f[0] = 0;
        check_exp(fx, f, n);
    }
    for (int lg = 7; lg <= kLgMax; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (std::size_t m : {n - 1, n, n + 1, n / 2 + pick(n / 2) + 1}) {
            if (m > (std::size_t(1) << kLgMax)) continue;
            auto f = random_poly(m, int(pick(3)));
            f[0] = 0;
            check_exp(fx, f, m, lg % 2 == 0);
        }
    }
}

// g = log(f) mod x^n by i g_i = i f_i - sum_(0<k<i) k g_k f_(i-k), f[0] = 1.
std::vector<u32> log_reference(const std::vector<u32>& f, std::size_t n) {
    const auto coefficient = [&f](std::size_t i) { return i < f.size() ? f[i] : 0; };
    std::vector<u32> g(n, 0), kg(n, 0);
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = mul(u32(i), coefficient(i));
        for (std::size_t k = 1; k < i; ++k) s = (s + P - u64(kg[k]) * coefficient(i - k) % P) % P;
        kg[i] = u32(s);
        g[i] = mul(kg[i], power(u32(i), P - 2));
    }
    return g;
}

void check_log(Fixture& fx, const std::vector<u32>& f, std::size_t n, bool in_place = false) {
    std::vector<u32> g(n, 0xFFFFFFFF);
    if (in_place) {
        g = f;
        g.resize(n);
        poly::log(fx.t, g, g, fx.log_scratch);
    } else {
        poly::log(fx.t, f, g, fx.log_scratch);
    }
    if (n <= 3000) {
        expect(g == log_reference(f, n), "log", n, f.size());
        return;
    }
    const std::size_t prefix = 1000;  // g mod x^m is log(f) mod x^m
    expect(std::equal(g.begin(), g.begin() + prefix, log_reference(f, prefix).begin()), "log prefix", n);
    expect(g[0] == 0, "log: g[0] = 0", n);
    const auto coefficient = [&f](std::size_t i) { return i < f.size() ? f[i] : 0; };
    const std::size_t k = (std::size_t(1) << poly::log_log(n)) / 2;  // q = f'/f in blocks of k
    std::vector<std::size_t> at = {n - 1, n - 2};
    for (std::size_t j = k; j < n - 1; j += k)  // g[i] = q[i - 1] / i
        for (std::size_t i = j; i < std::min(j + 3, n); ++i) at.push_back(i);
    for (int i = 0; i < 24; ++i) at.push_back(1 + pick(n - 1));
    for (std::size_t i : at) {  // i f_i = sum_k k g_k f_(i-k)
        u64 s = 0;
        for (std::size_t k = 1; k <= i; ++k) s = (s + u64(k) * g[k] % P * coefficient(i - k)) % P;
        expect(s == mul(u32(i), coefficient(i)), "f' = f g' at coefficient", n, i);
    }
}

void test_log(Fixture& fx) {
    for (std::size_t n = 1; n <= 160; ++n)
        for (int kind = 0; kind < 3; ++kind) {
            auto f = random_poly(n, kind);
            f[0] = 1;
            check_log(fx, f, n, kind == 1);
        }
    // Edge cases: f = 1 (g = 0), f = 1 - x (g_i = -1 / i), f shorter and longer than n; sizes
    // around the block boundaries (n - 1 = 3k, 3k + 1, 4k).
    for (std::size_t n : {65, 66, 1000, 3073, 3074, 4096, 4097, 4098, 70000}) {
        check_log(fx, {1}, n);
        check_log(fx, {1, P - 1}, n);
        auto f = random_poly(n / 3 + 1);
        f[0] = 1;
        check_log(fx, f, n);
        f = random_poly(2 * n);
        f[0] = 1;
        check_log(fx, f, n);
        f = random_poly(n);
        f[0] = 1;
        check_log(fx, f, n, true);
    }
    for (int lg = 7; lg <= kLgMax; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (std::size_t m : {n - 1, n, n + 1, n + 2, 3 * n / 4 + 1, 3 * n / 4 + 2, n / 2 + pick(n / 2) + 1}) {
            if (m > (std::size_t(1) << kLgMax) + 1) continue;
            auto f = random_poly(m, int(pick(3)));
            f[0] = 1;
            check_log(fx, f, m, lg % 2 == 0);
        }
    }
}


// Coefficient i of g = c (f / f[0])^e, from the coefficients below it: f g' = e f' g gives
// f[0] i g_i = sum_(0<j<=i) (e j - (i - j)) f_j g_(i-j).
u32 power_coefficient(const std::vector<u32>& f, u32 e, std::span<const u32> g, std::size_t i) {
    u64 s = 0;
    for (std::size_t j = 1; j <= i && j < f.size(); ++j) s = (s + u64(sub(mul(e, u32(j)), u32(i - j))) * f[j] % P * g[i - j]) % P;
    return u32(s);
}

std::vector<u32> power_reference(const std::vector<u32>& f, u32 e, u32 c, std::size_t n) {
    std::vector<u32> g(n);
    g[0] = c;
    const u32 inv0 = power(f[0], P - 2);
    for (std::size_t i = 1; i < n; ++i) g[i] = mul(mul(power_coefficient(f, e, g, i), power(u32(i), P - 2)), inv0);
    return g;
}

void check_power(Fixture& fx, const std::vector<u32>& f, u32 e, u32 c, std::size_t n, bool in_place = false) {
    std::vector<u32> g(n, 0xFFFFFFFF);
    if (in_place) {
        g = f;
        g.resize(n);
        poly::power(fx.t, g, e, c, g, fx.power_scratch);
    } else {
        poly::power(fx.t, f, e, c, g, fx.power_scratch);
    }
    if (n <= 3000) {
        expect(g == power_reference(f, e, c, n), "power", n, e);
        return;
    }
    const std::size_t prefix = 1000;  // g mod x^m is c (f / f[0])^e mod x^m
    expect(std::equal(g.begin(), g.begin() + prefix, power_reference(f, e, c, prefix).begin()), "power prefix", n, e);
    std::vector<std::size_t> at = {n - 1, n - 2, n / 2, n / 2 - 1};
    for (int i = 0; i < 24; ++i) at.push_back(1 + pick(n - 1));
    for (std::size_t i : at)
        expect(power_coefficient(f, e, g, i) == mul(mul(f[0], u32(i)), g[i]), "f g' = e f' g at coefficient", n, i);
}

void test_power(Fixture& fx) {
    const auto exponent = [](int trial) -> u32 { return trial % 4 == 0 ? 0 : trial % 4 == 1 ? 1 : trial % 4 == 2 ? (P + 1) / 2 : u32(pick(P)); };
    for (std::size_t n = 1; n <= 160; ++n)
        for (int kind = 0; kind < 4; ++kind) {
            auto f = random_poly(n, kind % 3);
            if (!f[0]) f[0] = 1 + u32(pick(P - 1));
            check_power(fx, f, exponent(kind + int(n)), u32(pick(P)), n, kind == 1);
        }
    // Edge cases: f = f[0] (g = c), f = 1 - x with e = -1 (g = all ones), f shorter and longer
    // than n, e = 1 (g = c f / f[0]), e = 0 (g = c); sizes around log's blocks.
    for (std::size_t n : {65, 66, 1000, 3073, 4096, 4097, 70000}) {
        check_power(fx, {5}, u32(pick(P)), 3, n);
        check_power(fx, {1, P - 1}, P - 1, 1, n);
        auto f = random_poly(n / 3 + 1);
        f[0] = 1 + u32(pick(P - 1));
        check_power(fx, f, u32(pick(P)), u32(pick(P)), n);
        f = random_poly(2 * n);
        f[0] = 1 + u32(pick(P - 1));
        check_power(fx, f, 1, 1, n);
        check_power(fx, f, 0, 7, n);
        f = random_poly(n);
        f[0] = 1 + u32(pick(P - 1));
        check_power(fx, f, u32(pick(P)), u32(pick(P)), n, true);
    }
    for (int lg = 7; lg <= kLgMax; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (std::size_t m : {n - 1, n, n + 1, 3 * n / 4 + 1, n / 2 + pick(n / 2) + 1}) {
            if (m > (std::size_t(1) << kLgMax)) continue;
            auto f = random_poly(m, int(pick(3)));
            if (!f[0]) f[0] = 1;
            check_power(fx, f, exponent(lg), u32(pick(P)), m, lg % 2 == 0);
        }
    }
}

// g = sqrt(f) mod x^n with g[0] = c by the recurrence 2 c g_i = f_i - sum_(0<j<i) g_j g_(i-j).
std::vector<u32> sqrt_reference(const std::vector<u32>& f, u32 c, std::size_t n) {
    std::vector<u32> g(n);
    g[0] = c;
    const u32 inverse = power(mul(2, c), P - 2);
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = i < f.size() ? f[i] : 0;
        for (std::size_t j = 1; j < i; ++j) s = (s + P - u64(g[j]) * g[i - j] % P) % P;
        g[i] = mul(u32(s), inverse);
    }
    return g;
}

void check_sqrt(Fixture& fx, const std::vector<u32>& f, u32 c, std::size_t n) {
    std::vector<u32> g(n, 0xFFFFFFFF);
    poly::sqrt(fx.t, f, c, g, fx.sqrt_scratch);
    if (n <= 3000) {
        expect(g == sqrt_reference(f, c, n), "sqrt", n, f.size());
        return;
    }
    const std::size_t prefix = 1000;  // g mod x^m is sqrt(f) mod x^m
    expect(std::equal(g.begin(), g.begin() + prefix, sqrt_reference(f, c, prefix).begin()), "sqrt prefix", n);
    std::vector<std::size_t> at = {n - 1, n - 2, n / 2, n / 2 - 1, n / 2 + n / 4, n / 2 + n / 4 + 1};
    for (int i = 0; i < 24; ++i) at.push_back(1 + pick(n - 1));
    for (std::size_t i : at) expect(product_coefficient(g, g, i) == (i < f.size() ? f[i] : 0), "g^2 = f at coefficient", n, i);
}

// f with f[0] = c^2 for a random c != 0, and c or -c.
std::pair<std::vector<u32>, u32> random_square(std::size_t size, int kind) {
    auto f = random_poly(size, kind);
    const u32 c = 1 + u32(pick(P - 1));
    f[0] = mul(c, c);
    return {f, pick(2) ? c : P - c};
}

void test_sqrt(Fixture& fx) {
    for (std::size_t n = 1; n <= 160; ++n)
        for (int kind = 0; kind < 3; ++kind) {
            const auto [f, c] = random_square(n, kind);
            check_sqrt(fx, f, c, n);
        }
    // Edge cases: f = c^2 (g = c), f = (1 - x)^2 (g = 1 - x), f shorter and longer than n; sizes
    // where the last step needs one or two products (n - m = m/2, m/2 + 1 for m < n <= 2m).
    for (std::size_t n : {65, 96, 97, 1000, 3000, 3072, 3073, 4096, 4097, 70000}) {
        check_sqrt(fx, {9}, P - 3, n);
        check_sqrt(fx, {1, P - 2, 1}, 1, n);
        for (std::size_t size : {n / 3 + 1, 2 * n}) {
            const auto [f, c] = random_square(size, 0);
            check_sqrt(fx, f, c, n);
        }
    }
    for (int lg = 7; lg <= kLgMax; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (std::size_t m : {n - 1, n, n + 1, 3 * n / 4, 3 * n / 4 + 1, n / 2 + pick(n / 2) + 1}) {
            if (m > (std::size_t(1) << kLgMax)) continue;
            const auto [f, c] = random_square(m, int(pick(3)));
            check_sqrt(fx, f, c, m);
        }
    }
}

}  // namespace

int main() {
    static Fixture fx;
    test_leaf_kernels(fx);
    test_transforms(fx);
    test_inverse(fx);
    test_derivative();
    test_divide_by_index();
    test_exp(fx);
    test_log(fx);
    test_power(fx);
    test_sqrt(fx);
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
}
