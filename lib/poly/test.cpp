// Tests for lib/poly against O(n^2) references: transforms leaf by leaf against their definition,
// products against schoolbook multiplication, the inverse, exp, log, power and sqrt against their recurrences,
// composition against Horner's rule and identities,
// coefficient-wise operations against scalar code. Long results are checked at random
// coefficients (each an O(n) sum).
#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "lib/poly/calculus.hpp"
#include "lib/poly/composition.hpp"
#include "lib/poly/compositional_inverse.hpp"
#include "lib/poly/divider.hpp"
#include "lib/poly/exp.hpp"
#include "lib/poly/holonomic.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/log.hpp"
#include "lib/poly/pow.hpp"
#include "lib/poly/projection.hpp"
#include "lib/poly/sparse.hpp"
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
                // Times c: 1 (the default), P - 1, random; with each output half.
                const u32 c = trial < 2 ? 1 : trial % 2 ? P - 1 : u32(rng() % P);
                if (c == 1) fx.t.cyclic_product(in, shift, out, tb, half);
                else fx.t.cyclic_product(in, shift, out, tb, half, c);
                for (std::size_t i : {lo, hi - 1, lo + pick(hi - lo), lo + pick(hi - lo)})
                    expect(out[i] == mul(c, cyclic_coefficient(shifted, b, i)), "cyclic_product of x^shift in, times c", lg, i);
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

// g = 1 / f mod x^n by inverse(), or (last_step_in_place) to k = 2^(inverse_log(n) - 1) by
// inverse(), then inverse_step() in a buffer that holds f and receives g[k, n).
void check_inverse(Fixture& fx, const std::vector<u32>& f, std::size_t n, bool last_step_in_place = false) {
    std::vector<u32> g(n, 0xFFFFFFFF);
    const std::size_t k = (std::size_t(1) << poly::inverse_log(n)) / 2;
    if (last_step_in_place && n > k && f.size() <= n) {
        const auto work = fx.load(0, f);
        poly::inverse(fx.t, work, std::span(g).first(k), fx.scratch);
        poly::inverse_step(fx.t, work, std::span(g).first(k), work.subspan(k, n - k), fx.buffer[1].first(2 * k),
                           fx.buffer[0].first(2 * k));
        std::copy_n(work.begin() + k, n - k, g.begin() + k);
    } else {
        poly::inverse(fx.t, f, g, fx.scratch);
    }
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
            check_inverse(fx, f, m, true);
        }
    }
    for (std::size_t n = 33; n <= 300; ++n) {
        auto f = random_poly(n - pick(3), int(pick(3)));
        if (!f[0]) f[0] = 1;
        check_inverse(fx, f, n, true);
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
        // the last step with and without its upper parts: n - m = m/2 and m/2 + 1
        for (std::size_t m : {n - 1, n, n + 1, 3 * n / 4, 3 * n / 4 + 1, n / 2 + pick(n / 2) + 1}) {
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

// exp and power with the smallest tables they allow (lg_max = exp_log(n), power_log(n)) against
// the fixture's (lg_max = kLgMax).
void test_minimal_tables(Fixture& fx) {
    for (std::size_t n : {1, 2, 64, 65, 96, 97, 127, 128, 129, 1000, 4096, 4097, 6145, 70000, 1 << 18, (1 << 18) + 1}) {
        auto f = random_poly(n);
        f[0] = 0;
        std::vector<u32> expected(n), got(n);
        poly::exp(fx.t, f, expected, fx.exp_scratch);
        {
            poly::Arena arena(poly::Transform::words(poly::exp_log(n)) + poly::exp_scratch(n));
            const poly::Transform t(arena, poly::exp_log(n));
            poly::exp(t, f, got, arena.take(poly::exp_scratch(n)));
        }
        expect(got == expected, "exp with lg_max = exp_log(n)", n);
        f[0] = 1 + u32(pick(P - 1));
        const u32 e = u32(pick(P)), c = u32(pick(P));
        poly::power(fx.t, f, e, c, expected, fx.power_scratch);
        {
            poly::Arena arena(poly::Transform::words(poly::power_log(n)) + poly::power_scratch(n));
            const poly::Transform t(arena, poly::power_log(n));
            poly::power(t, f, e, c, got, arena.take(poly::power_scratch(n)));
        }
        expect(got == expected, "power with lg_max = power_log(n)", n);
    }
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

// sparse::detail::reduce on 8 sums each, up to its bounds: 12 (P - 1)^2 unfolded, 2^64 - 1 folded.
template <bool kFold>
void check_sparse_reduce() {
    const u64 max = kFold ? ~u64(0) : poly::sparse::detail::kUnfolded * u64(P - 1) * (P - 1);
    const u32 unscale = poly::sparse::inverse(poly::sparse::detail::kR);  // 2^-32 mod P
    std::vector<u64> values = {0, 1, P - 1, P, max, max - 1, max - P, max / 2, u64(P) << 32, (u64(P) << 32) - 1};
    for (int i = 0; i < 4000; ++i) values.push_back(i % 2 ? max - rng() % (u64(1) << 40) : rng() % max);
    for (std::size_t i = 0; i + 8 <= values.size(); i += 8) {
        const u64* x = values.data() + i;
        const __m256i even = _mm256_setr_epi64x(x[0], x[2], x[4], x[6]), odd = _mm256_setr_epi64x(x[1], x[3], x[5], x[7]);
        alignas(32) u32 got[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(got), poly::sparse::detail::reduce<kFold>(even, odd));
        for (int k = 0; k < 8; ++k) expect(got[k] == mul(u32(x[k] % P), unscale), "sparse reduce", x[k], kFold);
    }
}

using poly::sparse::Recurrence;
using Terms = std::vector<poly::sparse::Term>;

// g[i] = r[i] + sum over taps (d, c) of c g[i - d], i < n.
std::vector<u32> recurrence_reference(const Terms& taps, const Terms& rhs, std::size_t n) {
    std::vector<u32> g(n, 0);
    for (const auto [i, r] : rhs)
        if (i < n) g[i] = r;
    for (std::size_t i = 0; i < n; ++i)
        for (const auto [d, c] : taps)
            if (d <= i) g[i] = add(g[i], mul(c, g[i - d]));
    return g;
}

// The recurrence solved by next() calls of random lengths (multiples of kBlock but the last):
// into one array, and, when the taps reach back at most `chunk`, into a ring of history() +
// chunk words whose last history() words move to its front after each call.
void check_recurrence(const Terms& taps, const Terms& rhs, std::size_t n) {
    const auto want = recurrence_reference(taps, rhs, n);
    const std::size_t padded = (n + Recurrence::kBlock - 1) / Recurrence::kBlock * Recurrence::kBlock;
    {
        Recurrence recurrence(taps, rhs);
        std::vector<u32> g(Recurrence::kPadding + padded, 0);
        for (std::size_t i = 0; i < n;) {
            const std::size_t m = std::min(n - i, Recurrence::kBlock * (1 + pick(40)));
            recurrence.next(g.data() + Recurrence::kPadding + i, m);
            i += m;
        }
        expect(std::equal(want.begin(), want.end(), g.begin() + Recurrence::kPadding), "sparse recurrence, array", n,
               taps.size());
    }
    const std::size_t chunk = Recurrence::kBlock * (1 + pick(64));
    Recurrence recurrence(taps, rhs);
    const std::size_t history = recurrence.history();
    if (history > chunk) return;
    std::vector<u32> ring(history + chunk, 0), got;
    for (std::size_t i = 0; i < n; i += chunk) {
        const std::size_t m = std::min(chunk, n - i);
        recurrence.next(ring.data() + history, m);
        got.insert(got.end(), ring.begin() + history, ring.begin() + history + m);
        std::copy(ring.begin() + chunk, ring.end(), ring.begin());
    }
    expect(got == want, "sparse recurrence, ring", n, taps.size());
}

// count distinct distances from pool, coefficients random or P - 1.
Terms random_taps(std::vector<u32> pool, std::size_t count, bool worst) {
    std::shuffle(pool.begin(), pool.end(), rng);
    Terms taps;
    for (std::size_t k = 0; k < std::min(count, pool.size()); ++k)
        taps.push_back({pool[k], worst ? P - 1 : u32(pick(P))});
    return taps;
}

std::vector<u32> range(u32 first, u32 last) {
    std::vector<u32> v(last - first);
    for (u32 i = first; i < last; ++i) v[i - first] = i;
    return v;
}

void test_recurrence() {
    check_sparse_reduce<false>();
    check_sparse_reduce<true>();
    const std::vector<u32> edges = {1, 2, 7, 8, 12, 13, 14, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100, 1000};
    for (int round = 0; round < 3000; ++round) {
        const std::size_t n = round < 200 ? std::size_t(round + 1) : 1 + pick(round % 10 ? 700 : 5000);
        const std::size_t count = round % 7 == 0 ? Recurrence::kMaxTaps : 1 + pick(10);
        const bool worst = round % 5 == 0;
        Terms taps;
        switch (round % 5) {
            case 0: taps = random_taps(range(1, 16), count, worst); break;     // short only
            case 1: taps = random_taps(range(16, 3000), count, worst); break;  // long only
            case 2: taps = random_taps(range(1, 200), count, worst); break;    // mixed
            case 3: taps = random_taps(edges, count, worst); break;
            default: taps = {}; break;
        }
        Terms rhs;  // distinct indices
        const bool at_zero = round % 3 == 0;
        if (at_zero) rhs.push_back({0, worst ? P - 1 : u32(pick(P))});
        if (round % 4 != 0) {
            const Terms more = random_taps(range(at_zero, u32(std::min<std::size_t>(n, 3000))), 1 + pick(4), worst);
            rhs.insert(rhs.end(), more.begin(), more.end());
        }
        std::ranges::sort(rhs, {}, &poly::sparse::Term::index);
        check_recurrence(taps, rhs, n);
    }
    // Each width, 1 .. 15, over a long range; and the widest sums: 15 short taps or 16 long or
    // mixed taps at P - 1.
    for (u32 w = 1; w < 16; ++w) {
        Terms taps = {{w, u32(pick(P))}};
        if (w > 1) taps.push_back({1, u32(pick(P))});
        check_recurrence(taps, {{0, 1}}, 20000);
    }
    check_recurrence(random_taps(range(1, 16), 15, true), {{0, P - 1}}, 20000);
    check_recurrence(random_taps(range(16, 40), 16, true), {{0, P - 1}, {5, P - 1}}, 20000);
    check_recurrence(random_taps(range(1, 40), 16, true), {{0, P - 1}, {30, P - 1}}, 20000);
}

// h = f(g) mod x^n by Horner's rule.
std::vector<u32> compose_reference(const std::vector<u32>& f, const std::vector<u32>& g, std::size_t n) {
    std::vector<u32> h(n, 0);
    for (std::size_t i = std::min(f.size(), n); i-- > 0;) {
        std::vector<u32> next(n, 0);
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = 1; b < g.size() && a + b < n; ++b) next[a + b] = u32((next[a + b] + u64(h[a]) * g[b]) % P);
        next[0] = f[i];
        h = std::move(next);
    }
    return h;
}

constexpr std::size_t kComposeMax = std::size_t(1) << 17;

// poly::compose with its own arena (tables and scratch for up to kComposeMax coefficients); the
// scratch is filled with garbage first.
std::vector<u32> compose(const std::vector<u32>& f, const std::vector<u32>& g, std::size_t n) {
    static poly::Arena arena(poly::Transform::words(poly::compose_log(kComposeMax)) + poly::compose_scratch(kComposeMax) + 64);
    static const poly::Transform t(arena, poly::compose_log(kComposeMax));
    static const std::span<u32> scratch = arena.take(poly::compose_scratch(kComposeMax));
    std::vector<u32> h(n, 0xFFFFFFFF);
    std::fill(scratch.begin(), scratch.end(), 0xFFFFFFFF);
    poly::compose(t, f, g, h, scratch);
    return h;
}

// Random g with g[0] = 0; kind 3: a random number of leading zeros.
std::vector<u32> random_inner(std::size_t n, int kind) {
    auto g = random_poly(n, kind % 3);
    if (kind == 3) std::fill(g.begin(), g.begin() + std::ptrdiff_t(pick(n)), 0);
    if (!g.empty()) g[0] = 0;
    return g;
}

// Long results by identities, at coefficients i (O(n) each): f = sum c^i y^i gives h = 1 / (1 - c g),
// so h - c h g = 1; the chain rule h' = (f' o g) g'; f = y^2 gives g^2.
void check_compose_identities(const std::vector<u32>& g, std::size_t n) {
    std::vector<std::size_t> at = {0, 1, 2, n - 1, n - 2, n / 2, n / 2 - 1};
    for (int i = 0; i < 20; ++i) at.push_back(pick(n));
    const u32 c = u32(pick(P));
    std::vector<u32> geometric(n);
    for (std::size_t i = 0, ci = 1; i < n; ++i, ci = mul(u32(ci), c)) geometric[i] = u32(ci);
    const auto h = compose(geometric, g, n);
    for (std::size_t i : at)
        expect(sub(h[i], mul(c, product_coefficient(h, g, i))) == (i == 0 ? 1 : 0), "compose: 1 / (1 - c g)", n, i);

    const auto f = random_poly(n);
    std::vector<u32> df(n - 1), dg(g.size() - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) df[i] = mul(u32(i + 1), f[i + 1]);
    for (std::size_t i = 0; i + 1 < g.size(); ++i) dg[i] = mul(u32(i + 1), g[i + 1]);
    const auto h1 = compose(f, g, n), h2 = compose(df, g, n);
    for (std::size_t i : at)
        if (i + 1 < n) expect(mul(u32(i + 1), h1[i + 1]) == product_coefficient(h2, dg, i), "compose: chain rule", n, i);

    const auto h3 = compose({0, 0, 1}, g, n);
    for (std::size_t i : at) expect(h3[i] == product_coefficient(g, g, i), "compose: y^2", n, i);
}

void test_compose() {
    for (std::size_t n = 1; n <= 160; ++n)
        for (int kind = 0; kind < 4; ++kind) {
            const auto f = random_poly(n, (kind + 1) % 3), g = random_inner(n, kind);
            expect(compose(f, g, n) == compose_reference(f, g, n), "compose", n, kind);
        }
    // f and g shorter and longer than n; g = 0 (h = f[0]) and g = x (h = f).
    for (std::size_t n : {33, 63, 64, 65, 200, 256, 257}) {
        for (std::size_t size : {std::size_t(1), std::size_t(2), n / 3 + 1, 2 * n}) {
            const auto f = random_poly(size), g = random_inner(n / 2 + 2, 0);
            const auto f2 = random_poly(n / 2 + 2), g2 = random_inner(size, 0);
            expect(compose(f, g, n) == compose_reference(f, g, n), "compose: f size", n, size);
            expect(compose(f2, g2, n) == compose_reference(f2, g2, n), "compose: g size", n, size);
        }
        const auto f = random_poly(n);
        std::vector<u32> constant(n, 0), x(n, 0);
        constant[0] = f[0], x[1] = 1;
        expect(compose(f, std::vector<u32>(n, 0), n) == constant, "compose: g = 0", n);
        expect(compose(f, x, n) == f, "compose: g = x", n);
    }
    for (int lg = 6; lg <= 17; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (std::size_t m : {n - 1, n, n + 1})
            if (m <= kComposeMax) check_compose_identities(random_inner(m, int(pick(4))), m);
    }
}

using poly::sparse::Holonomic;
using Taps = std::vector<poly::sparse::Tap>;

// 1 / i mod P for i < size, by the classic recurrence.
std::vector<u32> reciprocal_table(std::size_t size) {
    std::vector<u32> inv(std::max<std::size_t>(size, 2), 1);
    for (std::size_t i = 2; i < size; ++i) inv[i] = mul(P - P / u32(i), inv[P % i]);
    return inv;
}

// holonomic's two reduction steps at their bounds: partial on sums up to 12 (P - 1)^2 unfolded and
// 2^64 - 1 folded (a value below 4P in the low dword); scale on w < 4P and y < P.
template <bool kFold>
void check_holonomic_reduce() {
    namespace detail = poly::sparse::detail;
    const u64 max = kFold ? ~u64(0) : detail::kUnfolded * u64(P - 1) * (P - 1);
    const u32 unscale = poly::sparse::inverse(detail::kR);  // 2^-32 mod P
    std::vector<u64> values = {0, 1, P - 1, P, max, max - 1, max - P, max / 2, u64(P) << 32, (u64(P) << 32) - 1};
    for (int i = 0; i < 4000; ++i) values.push_back(i % 2 ? max - rng() % (u64(1) << 40) : rng() % max);
    for (std::size_t i = 0; i + 4 <= values.size(); i += 4) {
        const u64* x = values.data() + i;
        alignas(32) u64 got[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(got), detail::partial<kFold>(_mm256_setr_epi64x(x[0], x[1], x[2], x[3])));
        for (int k = 0; k < 4; ++k)
            expect(got[k] < 4 * u64(P) && got[k] % P == mul(u32(x[k] % P), unscale), "holonomic partial", x[k], kFold);
    }
}

// scale<kTerms>: w y + s with w < 4P, y < P and s up to kTerms (P - 1)^2.
template <std::size_t kTerms>
void check_holonomic_scale() {
    namespace detail = poly::sparse::detail;
    const u32 unscale = poly::sparse::inverse(detail::kR);
    const u64 edges[] = {0, 1, P - 1, P, 2 * u64(P), 4 * u64(P) - 1};
    const u64 max = kTerms * u64(P - 1) * (P - 1);
    for (int i = 0; i < 4000; ++i) {
        u64 w[4], y[4], s[4];
        for (int k = 0; k < 4; ++k) {
            w[k] = i % 3 ? rng() % (4 * u64(P)) : edges[rng() % 6];
            y[k] = i % 5 ? rng() % P : P - 1;
            s[k] = kTerms == 0 ? 0 : i % 4 == 0 ? max - rng() % (u64(1) << 40) : i % 7 == 0 ? max : rng() % max;
        }
        alignas(32) u64 got[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(got),
                           detail::scale<kTerms>(_mm256_setr_epi64x(w[0], w[1], w[2], w[3]), _mm256_setr_epi64x(y[0], y[1], y[2], y[3]),
                                                 _mm256_setr_epi64x(s[0], s[1], s[2], s[3])));
        for (int k = 0; k < 4; ++k) {
            const u32 want = mul(add(mul(u32(w[k] % P), u32(y[k])), u32(s[k] % P)), unscale);
            expect(got[k] == want, "holonomic scale", w[k] * 100 + kTerms, s[k]);
        }
    }
}

void test_inverses() {
    const u32 r = u32((u64(1) << 32) % P);  // Montgomery form of 1
    for (const auto& [first, count] : {std::pair<u32, std::size_t>{1, 0}, {1, 1}, {1, 31}, {1, 32}, {1, 33}, {2, 1000},
                                      {999983, 4097}, {P - 2000, 1900}}) {
        std::vector<u32> y(count + 32);
        poly::sparse::inverses(first, count, y.data());
        for (std::size_t i = 0; i < count; ++i)
            expect(mul(y[i], u32(first + i)) == r, "sparse inverses", first, i);
    }
}

// n g[n] = sum over taps (d, a, b) of (a + b n) g[n - d], g[0] = initial, n < size.
std::vector<u32> holonomic_reference(const Taps& taps, u32 initial, std::size_t size) {
    static const auto inv = reciprocal_table(1 << 17);
    std::vector<u32> g(size, 0);
    if (size) g[0] = initial;
    for (std::size_t i = 1; i < size; ++i) {
        u32 sum = 0;
        for (const auto [d, a, b] : taps)
            if (d <= i) sum = add(sum, mul(add(a, mul(b, u32(i))), g[i - d]));
        g[i] = mul(sum, inv[i]);
    }
    return g;
}

// As check_recurrence: next() calls of random lengths into one array, and into a ring.
void check_holonomic(const Taps& taps, u32 initial, std::size_t n) {
    const auto want = holonomic_reference(taps, initial, n);
    const std::size_t padded = (n + Holonomic::kBlock - 1) / Holonomic::kBlock * Holonomic::kBlock;
    {
        Holonomic recurrence(taps, initial, n);
        std::vector<u32> g(Holonomic::kPadding + padded, 0);
        for (std::size_t i = 0; i < n;) {
            const std::size_t m = std::min(n - i, Holonomic::kBlock * (1 + pick(40)));
            recurrence.next(g.data() + Holonomic::kPadding + i, m);
            i += m;
        }
        expect(std::equal(want.begin(), want.end(), g.begin() + Holonomic::kPadding), "holonomic, array", n, taps.size());
    }
    const std::size_t chunk = Holonomic::kBlock * (1 + pick(64));
    Holonomic recurrence(taps, initial, n);
    const std::size_t history = recurrence.history();
    if (history > chunk) return;
    std::vector<u32> ring(history + chunk, 0), got;
    for (std::size_t i = 0; i < n; i += chunk) {
        const std::size_t m = std::min(chunk, n - i);
        recurrence.next(ring.data() + history, m);
        got.insert(got.end(), ring.begin() + history, ring.begin() + history + m);
        std::copy(ring.begin() + chunk, ring.end(), ring.begin());
    }
    expect(got == want, "holonomic, ring", n, taps.size());
}

// count distinct distances from pool; constants and slopes random or P - 1, slopes zero if !slope.
Taps random_holonomic_taps(std::vector<u32> pool, std::size_t count, bool worst, bool slope) {
    Taps taps;
    for (const auto [d, c] : random_taps(std::move(pool), count, worst))
        taps.push_back({d, c, slope ? (worst ? P - 1 : u32(pick(P))) : 0});
    return taps;
}

void test_holonomic() {
    check_holonomic_reduce<false>();
    check_holonomic_reduce<true>();
    check_holonomic_scale<0>();
    check_holonomic_scale<1>();
    check_holonomic_scale<8>();
    check_holonomic_scale<9>();
    check_holonomic_scale<10>();
    check_holonomic_scale<11>();
    check_holonomic_scale<15>();
    test_inverses();
    const std::vector<u32> edges = {1, 2, 7, 8, 12, 13, 14, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100, 1000};
    for (int round = 0; round < 2000; ++round) {
        const std::size_t n = round < 200 ? std::size_t(round + 1) : 1 + pick(round % 10 ? 700 : 5000);
        const std::size_t count = round % 7 == 0 ? Holonomic::kMaxTaps : 1 + pick(10);
        const bool worst = round % 5 == 0, slope = round % 3 != 0;
        Taps taps;
        switch (round % 5) {
            case 0: taps = random_holonomic_taps(range(1, 16), count, worst, slope); break;     // short only
            case 1: taps = random_holonomic_taps(range(16, 3000), count, worst, slope); break;  // long only
            case 2: taps = random_holonomic_taps(range(1, 200), count, worst, slope); break;    // mixed
            case 3: taps = random_holonomic_taps(edges, count, worst, slope); break;
            default: taps = {}; break;
        }
        const u32 initial = round % 11 == 0 ? 0 : worst ? P - 1 : u32(pick(P));
        check_holonomic(taps, initial, n);
    }
    // Each width, 1 .. 15, with and without slopes, over a long range; the widest sums: 15 short
    // taps at P - 1 (16 products with slopes), 16 long or mixed taps.
    for (u32 w = 1; w < 16; ++w)
        for (const bool slope : {false, true}) {
            Taps taps = {{w, u32(pick(P)), slope ? u32(pick(P)) : 0}};
            if (w > 1) taps.push_back({1, u32(pick(P)), 0});
            check_holonomic(taps, 1, 20000);
        }
    check_holonomic(random_holonomic_taps(range(1, 16), 15, true, true), P - 1, 20000);
    check_holonomic(random_holonomic_taps(range(1, 16), 15, true, false), P - 1, 20000);
    check_holonomic(random_holonomic_taps(range(16, 40), 16, true, true), P - 1, 20000);
    check_holonomic(random_holonomic_taps(range(1, 40), 16, true, true), P - 1, 20000);
    // Across several windows of odd reciprocals (and the end of the table of the first half), with
    // long taps reaching blocks at and next to window edges.
    const u32 window = u32(Holonomic::kWindow);
    for (const std::size_t n : {std::size_t(3 * window + 100), std::size_t(5 * window)}) {
        check_holonomic(random_holonomic_taps(range(1, 16), 6, false, true), u32(pick(P)), n);
        check_holonomic({{3, u32(pick(P)), u32(pick(P))}, {window - 16, u32(pick(P)), u32(pick(P))},
                         {2 * window + 5, u32(pick(P)), 0}}, 1, n);
    }
}

// a[k] = [x^(n-1)] g^k from the powers of g.
std::vector<u32> projection_reference(const std::vector<u32>& g, std::size_t n) {
    std::vector<u32> a(n), power(n, 0);
    power[0] = 1;
    const std::span<const u32> head = std::span<const u32>(g).first(std::min(g.size(), n));
    for (std::size_t k = 0; k < n; ++k) {
        a[k] = power[n - 1];
        std::vector<u32> next(n, 0);
        for (std::size_t i = 0; i < n; ++i) next[i] = product_coefficient(power, head, i);
        power = std::move(next);
    }
    return a;
}

// poly::power_projection with its own arena, as compose() above.
std::vector<u32> power_projection(const std::vector<u32>& g, std::size_t n) {
    static poly::Arena arena(poly::Transform::words(poly::projection_log(kComposeMax)) + poly::projection_scratch(kComposeMax) + 64);
    static const poly::Transform t(arena, poly::projection_log(kComposeMax));
    static const std::span<u32> scratch = arena.take(poly::projection_scratch(kComposeMax));
    std::vector<u32> a(n, 0xFFFFFFFF);
    std::fill(scratch.begin(), scratch.end(), 0xFFFFFFFF);
    poly::power_projection(t, g, a, scratch);
    return a;
}

u32 dot(const std::vector<u32>& a, const std::vector<u32>& b) {
    u64 s = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) s = (s + u64(a[i]) * b[i]) % P;
    return u32(s);
}

// Long results by transposition: sum_k f[k] a[k] = (f o g)[n - 1] for random f, and for f = y^k.
void check_projection_transposed(const std::vector<u32>& g, std::size_t n) {
    const auto a = power_projection(g, n);
    for (int trial = 0; trial < 3; ++trial) {
        const auto f = random_poly(n, trial % 3);
        expect(dot(f, a) == compose(f, g, n)[n - 1], "power_projection: transpose of compose", n, trial);
    }
    for (std::size_t k : {std::size_t(0), std::size_t(1), n / 2, n - 1, pick(n)}) {
        std::vector<u32> f(k + 1, 0);
        f[k] = 1;
        expect(a[k] == compose(f, g, n)[n - 1], "power_projection: a[k]", n, k);
    }
}

void test_projection() {
    for (std::size_t n = 1; n <= 160; ++n)
        for (int kind = 0; kind < 4; ++kind) {
            const auto g = random_inner(n, kind);
            expect(power_projection(g, n) == projection_reference(g, n), "power_projection", n, kind);
        }
    // g shorter and longer than n; g = 0 (a = [n == 1]), g = c x (a = c^(n-1) at n - 1).
    for (std::size_t n : {33, 63, 64, 65, 200, 256, 257}) {
        for (std::size_t size : {std::size_t(1), std::size_t(2), n / 3 + 1, 2 * n}) {
            const auto g = random_inner(size, int(pick(4)));
            expect(power_projection(g, n) == projection_reference(g, n), "power_projection: g size", n, size);
        }
        const u32 c = u32(pick(P));
        std::vector<u32> zero(n, 0), line(n, 0), want(n, 0);
        line[1] = c, want[n - 1] = power(c, n - 1);
        expect(power_projection(zero, n) == zero, "power_projection: g = 0", n);
        expect(power_projection(line, n) == want, "power_projection: g = c x", n);
    }
    for (int lg = 6; lg <= 17; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (std::size_t m : {n - 1, n, n + 1, n / 2 + 1 + pick(n / 2)})
            if (m <= kComposeMax) check_projection_transposed(random_inner(m, int(pick(4))), m);
    }
}

// g with g(f) = x mod x^n, coefficient by coefficient from the powers of f.
std::vector<u32> compositional_inverse_reference(const std::vector<u32>& f, std::size_t n) {
    std::vector<std::vector<u32>> powers(n, std::vector<u32>(n, 0));  // f^j mod x^n
    powers[0][0] = 1;
    const std::span<const u32> head = std::span<const u32>(f).first(std::min(f.size(), n));
    for (std::size_t j = 1; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i) powers[j][i] = product_coefficient(powers[j - 1], head, i);
    std::vector<u32> g(n, 0);
    for (std::size_t k = 1; k < n; ++k) {
        u32 s = k == 1;
        for (std::size_t j = 1; j < k; ++j) s = sub(s, mul(g[j], powers[j][k]));
        g[k] = mul(s, power(powers[k][k], P - 2));
    }
    return g;
}

constexpr std::size_t kInverseMax = std::size_t(1) << 17;

std::vector<u32> compositional_inverse(const std::vector<u32>& f, std::size_t n) {
    static const std::size_t words = poly::compositional_inverse_scratch(kInverseMax);
    static poly::Arena arena(poly::Transform::words(poly::compositional_inverse_log(kInverseMax)) + words + 64);
    static const poly::Transform t(arena, poly::compositional_inverse_log(kInverseMax));
    static const std::span<u32> scratch = arena.take(words);
    std::vector<u32> g(n, 0xFFFFFFFF);
    std::fill(scratch.begin(), scratch.end(), 0xFFFFFFFF);
    poly::compositional_inverse(t, f, g, scratch);
    return g;
}

// f with f[0] = 0 and f[1] != 0.
std::vector<u32> random_invertible(std::size_t n, int kind) {
    auto f = random_poly(std::max<std::size_t>(n, 2), kind);
    f[0] = 0;
    if (!f[1]) f[1] = 1 + u32(pick(P - 1));
    return f;
}

// Long results: f(g) = x and g(f) = x mod x^n, by compose.
void check_compositional_inverse(const std::vector<u32>& f, std::size_t n) {
    const auto g = compositional_inverse(f, n);
    std::vector<u32> x(n, 0);
    if (n > 1) x[1] = 1;
    expect(compose(f, g, n) == x, "compositional_inverse: f(g) = x", n);
    expect(compose(g, f, n) == x, "compositional_inverse: g(f) = x", n);
}

void test_compositional_inverse() {
    for (std::size_t n = 1; n <= 100; ++n)
        for (int kind = 0; kind < 3; ++kind) {
            const auto f = random_invertible(n, kind);
            expect(compositional_inverse(f, n) == compositional_inverse_reference(f, n), "compositional_inverse", n, kind);
        }
    // f = x (g = x), f = c x (g = x / c), f = x / (1 - x) (g = x / (1 + x)), f shorter than n.
    for (std::size_t n : {2, 3, 33, 64, 65, 1000, 4097}) {
        std::vector<u32> x(n, 0), geometric(n, 1), alternating(n, 0);
        x[1] = 1, geometric[0] = 0;
        for (std::size_t i = 1; i < n; ++i) alternating[i] = i % 2 ? 1 : P - 1;
        expect(compositional_inverse(x, n) == x, "compositional_inverse: f = x", n);
        expect(compositional_inverse(geometric, n) == alternating, "compositional_inverse: x / (1 - x)", n);
        check_compositional_inverse({0, 5}, n);
        check_compositional_inverse(random_invertible(n / 3 + 2, 0), n);
    }
    for (int lg = 6; lg <= 17; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (std::size_t m : {n - 1, n, n + 1})
            if (m <= kInverseMax) check_compositional_inverse(random_invertible(m, int(pick(3))), m);
    }
}

// Divider over [0, n) in calls of random multiples of kStep, into a separate array or in place;
// G random, or P - 1 (kind 1), or zero but at a few indices (kind 2).
void check_divider(std::size_t n, int kind, bool in_place) {
    using poly::sparse::Divider;
    static const auto inv = reciprocal_table(std::size_t(1) << 21);
    const std::size_t padded = (n + Divider::kStep - 1) / Divider::kStep * Divider::kStep;
    std::vector<u32> G(padded + 1), g(padded, P);
    for (std::size_t i = 0; i < padded + 1; ++i) G[i] = kind == 1 ? P - 1 : kind == 2 && pick(100) ? 0 : u32(pick(P));
    const std::vector<u32> input = G;
    Divider divider(n);
    for (std::size_t i = 0; i < n;) {
        const std::size_t m = std::min(n - i, Divider::kStep * (1 + pick(i < 4096 ? 8 : 600)));
        divider.divide(G.data() + i, in_place ? G.data() + i : g.data() + i, i, m);
        i += m;
    }
    const std::vector<u32>& got = in_place ? G : g;
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < n; ++i) wrong += got[i] != (i == 0 ? 0 : mul(input[i], inv[i]));
    expect(wrong == 0, "divider", n, wrong);
}

void test_divider() {
    for (std::size_t n = 1; n <= 300; ++n) check_divider(n, int(n % 3), n % 2);
    for (int round = 0; round < 60; ++round) check_divider(1 + pick(round % 4 ? 20000 : 300000), round % 3, round % 2);
    for (const std::size_t n : {std::size_t(1000000), std::size_t(1) << 20}) {
        check_divider(n, 0, false);
        check_divider(n, 1, true);
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
    test_minimal_tables(fx);
    test_sqrt(fx);
    test_recurrence();
    test_holonomic();
    test_divider();
    test_compose();
    test_projection();
    test_compositional_inverse();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
}
