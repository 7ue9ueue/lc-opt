// Tests for lib/poly against O(n^2) references: transforms leaf by leaf against their definition,
// products against schoolbook multiplication, the inverse (also bivariate), exp, log, power and sqrt
// against their recurrences, composition against Horner's rule and identities, product trees against
// naive products, chirps against their recurrence, multipoint evaluation and interpolation against
// Horner's rule, the Newton basis against synthetic division, division against long division,
// half-gcd jumps and inverses modulo a polynomial against the extended Euclidean algorithm,
// coefficient-wise operations against scalar code.
// Long results are checked at random coefficients (each an O(n) sum) or at random points.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/poly/calculus.hpp"
#include "lib/poly/chirp.hpp"
#include "lib/poly/composition.hpp"
#include "lib/poly/compositional_inverse.hpp"
#include "lib/poly/divider.hpp"
#include "lib/poly/division.hpp"
#include "lib/poly/evaluation.hpp"
#include "lib/poly/exp.hpp"
#include "lib/poly/factorials.hpp"
#include "lib/poly/gcd.hpp"
#include "lib/poly/holonomic.hpp"
#include "lib/poly/interpolation.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/inverse_2d.hpp"
#include "lib/poly/log.hpp"
#include "lib/poly/newton.hpp"
#include "lib/poly/pow.hpp"
#include "lib/poly/product_tree.hpp"
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

// times() with a factor per lane (Factors) against scalar products; the inverse scales.
void test_factors(Fixture& fx) {
    using poly::detail::Vec;
    for (int trial = 0; trial < 2000; ++trial) {
        alignas(32) u32 x[8], out[8], w[8];
        for (u32& v : x) v = trial % 4 == 0 ? ~u32(rng() % 3) : u32(rng());
        poly::detail::Factors f;
        if (trial % 2) {  // table entries
            const std::size_t k = 8 * pick(std::size_t(1) << (kLgMax - 7));
            f = poly::detail::entries(fx.roots.data(), k);
            for (int l = 0; l < 8; ++l) w[l] = fx.roots[ntt::detail::slot(k + l)];
        } else {
            for (u32& v : w) v = trial % 8 == 0 ? P - 1 : u32(rng() % P);
            alignas(32) u32 q[8];
            for (int l = 0; l < 8; ++l) q[l] = ntt::detail::quotient(w[l]);
            f = {poly::detail::load(w), poly::detail::load(q)};
        }
        poly::detail::store(out, poly::detail::times(poly::detail::load(x), f));
        for (int l = 0; l < 8; ++l) expect(out[l] < 2 * P && out[l] % P == mul(x[l] % P, w[l]), "times: per lane", trial, l);
    }
    for (int lg = 3; lg <= ntt::kMaxLog; ++lg) {
        const u32 s = poly::detail::kInverseScales[0][lg], sr = poly::detail::kInverseScales[1][lg];
        expect(mul(s, power(2, lg - 3)) == 1, "kInverseScales: (n / 8)^-1", lg);
        expect(sr == mul(s, u32((u64(1) << 32) % P)), "kInverseScales: times 2^32", lg);
    }
}

// The column kernels (pruned transforms) against forward_h1 and inverse_h1 on each column with
// bit b clear; the other columns unchanged. Inputs < 4P (forward) and < 2P (inverse).
void test_column_kernels(Fixture& fx) {
    using poly::detail::Vec;
    const std::span<u32> a = fx.buffer[0], want = fx.buffer[1];
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t h = std::size_t(4) << pick(7), b = pick(std::size_t(std::countr_zero(h)));
        const std::size_t k = trial % 5 == 0 ? 0 : pick(std::size_t(1) << (kLgMax - 6));
        const bool forward = trial % 2 == 0;
        const u32 bound = forward ? 4 * P : 2 * P;
        for (std::size_t i = 0; i < 32 * h; ++i) a[i] = trial % 7 == 0 ? bound - 1 - u32(rng() % 2) : u32(rng() % bound);
        std::copy_n(a.begin(), 32 * h, want.begin());
        const u32* table = forward ? fx.roots.data() : fx.t.inverse_roots();
        const poly::detail::Group w(table, k);
        for (std::size_t j = 0; j < h; ++j) {
            if (j >> b & 1) continue;
            u32* x = want.data() + 8 * j;
            Vec f[4] = {poly::detail::load(x), poly::detail::load(x + 8 * h), poly::detail::load(x + 16 * h),
                        poly::detail::load(x + 24 * h)};
            if (forward) poly::detail::forward_h1(f, w);
            else poly::detail::inverse_h1(f, w);
            for (std::size_t t = 0; t < 4; ++t) poly::detail::store(x + 8 * t * h, f[t]);
        }
        if (forward) poly::detail::forward_columns(a.data(), h, b, table, k);
        else poly::detail::inverse_columns(a.data(), h, b, table, k);
        expect(std::equal(a.begin(), a.begin() + std::ptrdiff_t(32 * h), want.begin()), "column kernels", h, b);
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
            // hold garbage); unaligned shifts and sizes. Output halves. Trials 6 and 7: in place
            // in one half with partial first and last vectors (forward_top8).
            for (int trial = 0; trial < 8; ++trial) {
                const std::size_t shift = trial == 0 ? 0 : trial == 1 || trial == 2 ? n / 2 : trial == 6 ? 3 : trial == 7 ? n / 2 + 5 : pick(n);
                const std::size_t size = trial == 0 || trial == 1 ? n / 2 : trial >= 6 ? n / 2 - 12 : pick(n - shift + 1);
                const bool in_place = trial % 2 == 1 || trial == 6;
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

// g = 1 / f mod (x^rows, y^cols), row-major, by f_00 g_ij = [i = j = 0] - sum_((p, q) != 0) f_pq g_(i-p)(j-q).
std::vector<u32> inverse_2d_reference(const std::vector<u32>& f, std::size_t rows, std::size_t cols) {
    std::vector<u32> g(rows * cols);
    const u32 inv0 = power(f[0], P - 2);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            u64 s = i == 0 && j == 0;
            for (std::size_t p = 0; p <= i; ++p)
                for (std::size_t q = 0; q <= j; ++q)
                    if (p || q) s = (s + u64(P - f[p * cols + q]) * g[(i - p) * cols + j - q]) % P;
            g[i * cols + j] = mul(u32(s), inv0);
        }
    return g;
}

// Coefficient (i, j) of f g, both row-major with cols columns.
u32 product_2d_coefficient(const std::vector<u32>& f, const std::vector<u32>& g, std::size_t cols, std::size_t i,
                           std::size_t j) {
    u64 s = 0;
    for (std::size_t p = 0; p <= i; ++p)
        for (std::size_t q = 0; q <= j; ++q) s = (s + u64(f[p * cols + q]) * g[(i - p) * cols + j - q]) % P;
    return u32(s);
}

// Inverse2d on its own arena (sizes from the plan), scratch filled with garbage first; against the
// recurrence up to 600 coefficients, else f g = 1 at the corners and random coefficients.
void check_inverse_2d(const std::vector<u32>& f, std::size_t rows, std::size_t cols,
                      poly::Inverse2d::Method method = poly::Inverse2d::Method::kAuto) {
    const poly::Inverse2d plan(rows, cols, method);
    using poly::Arena;
    Arena arena(poly::Transform::words(plan.lg()) + Arena::footprint(plan.f_words()) + Arena::footprint(plan.g_words()) +
                Arena::footprint(plan.scratch_words()));
    const poly::Transform t(arena, plan.lg());
    const auto fz = arena.take(plan.f_words()), gz = arena.take(plan.g_words()), scratch = arena.take(plan.scratch_words());
    std::fill(scratch.begin(), scratch.end(), 0xFFFFFFFF);
    const std::size_t stride = plan.stride(), split = plan.split(), offset = plan.offset();
    expect(stride >= cols && split >= (rows + 1) / 2 && split <= rows, "inverse_2d layout", rows, cols);
    for (std::size_t i = 0; i < rows; ++i) std::copy_n(f.begin() + i * cols, cols, fz.begin() + offset + i * stride);
    plan.run(t, fz, gz, scratch);
    std::vector<u32> g(rows * cols);
    for (std::size_t i = 0; i < rows; ++i)
        std::copy_n(i < split ? gz.begin() + i * stride : fz.begin() + offset + i * stride, cols, g.begin() + i * cols);
    if (rows * cols <= 600) {
        expect(g == inverse_2d_reference(f, rows, cols), "inverse_2d", rows, cols);
        return;
    }
    std::vector<std::pair<std::size_t, std::size_t>> at = {{0, 0}, {rows - 1, cols - 1}, {rows - 1, 0}, {0, cols - 1}};
    for (int k = 0; k < 8; ++k) at.push_back({pick(rows), pick(cols)});
    for (auto [i, j] : at)
        expect(product_2d_coefficient(f, g, cols, i, j) == (i == 0 && j == 0), "inverse_2d f g = 1", rows * cols, i * cols + j);
}

// Both methods (row by row where it applies) on a random f.
void check_inverse_2d(std::size_t rows, std::size_t cols, int kind) {
    auto f = random_poly(rows * cols, kind);
    if (!f[0]) f[0] = 1 + u32(pick(P - 1));
    check_inverse_2d(f, rows, cols, poly::Inverse2d::Method::kNewton);
    if (rows >= 3 && rows <= poly::Inverse2d::kMaxRowByRow) check_inverse_2d(f, rows, cols, poly::Inverse2d::Method::kRowByRow);
}

void test_inverse_2d() {
    for (std::size_t rows = 1; rows <= 12; ++rows)
        for (std::size_t cols = 1; cols <= 12; ++cols) check_inverse_2d(rows, cols, int((rows + cols) % 3));
    for (auto [rows, cols] : {std::pair<std::size_t, std::size_t>{1, 1000}, {2, 3000}, {3, 999}, {5, 600}, {9, 333}, {32, 40},
                              {33, 33}, {64, 64}, {65, 63}, {100, 7}, {255, 4}, {1023, 2}, {3000, 3}, {1000, 1}, {600, 5}})
        for (int kind = 0; kind < 3; ++kind) {
            check_inverse_2d(rows, cols, kind);
            check_inverse_2d(cols, rows, kind);
        }
    // 1 - x - y: g_ij = binomial(i + j, i); f = 1: g = 1.
    for (auto [rows, cols] : {std::pair<std::size_t, std::size_t>{20, 30}, {300, 400}}) {
        std::vector<u32> f(rows * cols);
        f[0] = 1, f[1] = P - 1, f[cols] = P - 1;
        check_inverse_2d(f, rows, cols);
        std::vector<u32> one(rows * cols);
        one[0] = 1;
        check_inverse_2d(one, rows, cols);
    }
    // The shapes of the Library Checker tests (N M <= 500000), and longer rows; the method the plan
    // picks, and both where row by row applies.
    for (auto [rows, cols] : {std::pair<std::size_t, std::size_t>{707, 707}, {1045, 478}, {478, 1045}, {10, 50000}, {53336, 9},
                              {9, 53336}, {6, 53336}, {2, 250000}, {250000, 2}, {1, 1 << 19}, {1 << 19, 1}, {3, 166666},
                              {32, 15625}, {5000, 100}}) {
        const int kind = int(pick(3));
        auto f = random_poly(rows * cols, kind);
        if (!f[0]) f[0] = 1 + u32(pick(P - 1));
        check_inverse_2d(f, rows, cols);
        if (rows >= 3 && rows <= 16) {
            check_inverse_2d(f, rows, cols, poly::Inverse2d::Method::kNewton);
            check_inverse_2d(f, rows, cols, poly::Inverse2d::Method::kRowByRow);
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

// sqrt_steps, the forward in place and sqrt_last_step writing g[m, n) over f[m, n), as the
// problem uses them, with f shift words into its buffer (any alignment); against poly::sqrt.
void check_sqrt_parts(Fixture& fx, const std::vector<u32>& f, u32 c, std::size_t shift) {
    const std::size_t n = f.size(), m = std::size_t(1) << poly::sqrt_log(n);
    std::vector<u32> expected(n);
    poly::sqrt(fx.t, f, c, expected, fx.sqrt_scratch);
    std::vector<u32> buffer(shift + n);
    std::copy(f.begin(), f.end(), buffer.begin() + std::ptrdiff_t(shift));
    const std::span<u32> u(buffer.data() + shift, n), a = fx.buffer[0].first(m), b = fx.buffer[1].first(m),
                         ht = fx.buffer[2].first(m);
    poly::sqrt_steps(fx.t, u, c, a, b, ht);
    const bool low = std::equal(a.begin(), a.end(), expected.begin());
    fx.t.forward(a);
    poly::sqrt_last_step(fx.t, u, a, ht, b, u.subspan(m));
    expect(low && std::equal(u.begin() + std::ptrdiff_t(m), u.end(), expected.begin() + std::ptrdiff_t(m)), "sqrt in parts",
           n, shift);
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
    // In parts, in place: one and two products in the last step, both top-level radices, f at
    // offsets 0 to 3 words.
    for (std::size_t n : {65, 96, 97, 128, 129, 200, 1000, 3072, 3073, 4096, 4097, 70000, 131073, 200001, 262144}) {
        for (std::size_t shift = 0; shift < 4; ++shift) {
            const auto [f, c] = random_square(n, int(pick(3)));
            check_sqrt_parts(fx, f, c, shift);
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

// As check_recurrence: next() calls of random lengths into one array, and into a ring held in the
// recurrence's spare words, with a canary after it.
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
    std::size_t history = Holonomic::kPadding;
    for (const auto& tap : taps) history = std::max<std::size_t>(history, tap.distance);
    constexpr std::size_t kCanary = 64;
    Holonomic recurrence(taps, initial, n, history + chunk + kCanary);
    expect(recurrence.history() == history, "holonomic, history", n, history);
    if (history > chunk) return;
    u32* const ring = recurrence.spare();
    expect(std::all_of(ring, ring + history + chunk + kCanary, [](u32 x) { return x == 0; }), "holonomic, spare zeros", n, 0);
    for (std::size_t i = 0; i < kCanary; ++i) ring[history + chunk + i] = u32(i * 2654435761u);
    std::vector<u32> got;
    for (std::size_t i = 0; i < n; i += chunk) {
        const std::size_t m = std::min(chunk, n - i);
        recurrence.next(ring + history, m);
        got.insert(got.end(), ring + history, ring + history + m);
        std::copy(ring + chunk, ring + chunk + history, ring);
    }
    expect(got == want, "holonomic, ring", n, taps.size());
    for (std::size_t i = 0; i < kCanary; ++i) expect(ring[history + chunk + i] == u32(i * 2654435761u), "holonomic, canary", n, i);
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
    check_holonomic(random_holonomic_taps(range(1, 9), 8, true, true), P - 1, 20000);  // the chained kernel's widest
    check_holonomic(random_holonomic_taps(range(1, 9), 8, true, false), P - 1, 20000);
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
    // The table of reciprocals is a ring from n = 2 kWindow on: n across the point where its reads
    // wrap (2 ring), with short and mixed taps. The random state is restored after them, so the
    // later tests keep their inputs.
    const std::mt19937_64 saved = rng;
    for (int round = 0; round < 6; ++round) {
        const std::size_t n = 2 * window + 1 + pick(6 * window);
        check_holonomic(random_holonomic_taps(range(1, 9), 1 + pick(8), round % 2, true), u32(pick(P)), n);
        check_holonomic(random_holonomic_taps(range(1, 3 * window), 1 + pick(10), false, round % 3 != 0), u32(pick(P)), n);
    }
    rng = saved;
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
// G random, or P - 1 (kind 1), or zero but at a few indices (kind 2). The table starts as garbage.
void check_divider(std::size_t n, int kind, bool in_place) {
    using poly::sparse::Divider;
    static const auto inv = reciprocal_table(std::size_t(1) << 21);
    const std::size_t padded = (n + Divider::kStep - 1) / Divider::kStep * Divider::kStep;
    std::vector<u32> G(padded + 1), g(padded, P);
    for (std::size_t i = 0; i < padded + 1; ++i) G[i] = kind == 1 ? P - 1 : kind == 2 && pick(100) ? 0 : u32(pick(P));
    const std::vector<u32> input = G;
    std::vector<u32> table(Divider::table_words(n));
    for (u32& x : table) x = u32(pick(P));
    Divider divider(n, table.data());
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

// Product trees (product_tree.hpp). Montgomery form: x 2^32 mod P.
u32 to_m(u32 x) { return u32((u64(x) << 32) % P); }
u32 from_m(u32 x) { return mul(x, power(to_m(1), P - 2)); }

std::vector<u32> naive_product(const std::vector<std::vector<u32>>& polys) {
    std::vector<u32> f{1};
    for (const auto& g : polys) {
        std::vector<u32> h(f.size() + g.size() - 1, 0);
        for (std::size_t i = 0; i < f.size(); ++i)
            for (std::size_t j = 0; j < g.size(); ++j) h[i + j] = add(h[i + j], mul(f[i], g[j]));
        f = h;
    }
    return f;
}

u32 evaluate(std::span<const u32> f, u32 x) {
    u32 v = 0;
    for (std::size_t i = f.size(); i-- > 0;) v = add(mul(v, x), f[i]);
    return v;
}

// A random polynomial of degree d with a nonzero leading coefficient; kind as random_poly.
std::vector<u32> random_degree(std::size_t d, int kind) {
    auto a = random_poly(d + 1, kind);
    if (a[d] == 0) a[d] = 1 + u32(rng() % (P - 1));
    return a;
}

// TreeTransform against Transform's definition, 8 words (one leaf) up to 2^15.
void test_tree_transform(Fixture& fx) {
    static poly::Arena arena(poly::TreeTransform::words(16) + 8 * poly::Arena::footprint(1 << 16));
    static const poly::TreeTransform t(arena, 16);
    static std::span<u32> buf[4] = {arena.take(1 << 16), arena.take(1 << 16), arena.take(1 << 16), arena.take(1 << 16)};
    for (int lg = 3; lg <= 15; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (int kind = 0; kind < 3; ++kind) {
            const auto a = random_poly(n, kind), b = random_poly(n, (kind + 1) % 3);
            const std::size_t shift = kind == 0 ? 0 : pick(n), size = kind == 0 ? n : pick(n - shift + 1);
            std::vector<u32> shifted(2 * n, 0);
            for (std::size_t i = 0; i < size; ++i) shifted[shift + i] = a[i];
            const std::vector<u32> low(shifted.begin(), shifted.begin() + std::ptrdiff_t(n));
            std::copy(a.begin(), a.begin() + std::ptrdiff_t(size), buf[1].begin());
            auto out = buf[0].first(n);
            t.forward(buf[1].first(size), shift, out);
            for (std::size_t p : leaves_to_check(n)) {
                const auto want = leaf(low, p, fx.roots.data());
                expect(std::equal(want.begin(), want.end(), out.begin() + 8 * p), "TreeTransform forward leaf", lg, p);
            }
            if (lg < 15) {
                auto upper = buf[2].first(n);
                t.forward_upper(buf[1].first(size), shift, upper);
                for (std::size_t p : leaves_to_check(n)) {
                    const auto want = leaf(shifted, n / 8 + p, fx.roots.data());
                    expect(std::equal(want.begin(), want.end(), upper.begin() + 8 * p), "TreeTransform forward_upper leaf", lg, p);
                }
            }
            const u32 c = kind == 2 ? P - 1 : kind == 1 ? u32(rng() % P) : 1;
            t.inverse(out, buf[3].first(n), c);
            bool ok = true;
            for (std::size_t i = 0; i < n; ++i) ok &= buf[3][i] == mul(c, low[i]);
            expect(ok, "TreeTransform inverse, times c", lg, kind);
            t.inverse(out, out);
            expect(equal(out, low), "TreeTransform inverse in place", lg, kind);

            // Products of transforms carry 2^-32: Montgomery-form inputs give Montgomery-form products.
            std::vector<u32> am(n), bm(n);
            for (std::size_t i = 0; i < n; ++i) am[i] = to_m(a[i]), bm[i] = to_m(b[i]);
            auto ta = fx.load(0, am), tb = fx.load(1, bm);
            t.forward(ta, 0, ta);
            t.forward(tb, 0, tb);
            t.leaf_products(ta.data(), tb.data(), ta);
            t.inverse(ta, ta);
            for (std::size_t i : {std::size_t(0), n - 1, pick(n), pick(n)})
                expect(from_m(ta[i]) == cyclic_coefficient(a, b, i), "TreeTransform leaf_products", lg, i);
            auto pa = fx.load(2, am), pb = fx.load(3, bm);
            poly::TreeTransform::pointwise_products(pa.data(), pb.data(), pa);
            for (std::size_t i : {std::size_t(0), n - 1, pick(n)}) expect(pa[i] == to_m(mul(a[i], b[i])), "pointwise_products", lg, i);
        }
    }
}

// multiply_lanes against schoolbook, every m + k <= 32, coefficients random, 0 or P - 1.
void test_multiply_lanes() {
    alignas(32) static u32 a[8 * 33], b[8 * 33], c[8 * 65];
    const u32 inverse_r = power(to_m(1), P - 2);
    for (int kind = 0; kind < 3; ++kind) {
        for (std::size_t m = 0; m <= 32; ++m) {
            for (std::size_t k = 0; m + k <= 32; ++k) {
                // kind 2: near P - 1, so the sums come close to their bound with varied low words
                const auto value = [kind] { return kind == 2 ? P - 1 - u32(rng() % 1024) : kind == 1 && rng() % 2 ? 0 : u32(rng() % P); };
                for (std::size_t i = 0; i < 8 * (m + 1); ++i) a[i] = value();
                for (std::size_t i = 0; i < 8 * (k + 1); ++i) b[i] = kind == 1 ? u32(rng() % P) : value();
                poly::detail::multiply_lanes_any(a, m, b, k, c);
                bool ok = true;
                for (std::size_t l = 0; l < 8; ++l)
                    for (std::size_t t = 0; t <= m + k; ++t) {
                        u32 s = 0;
                        for (std::size_t i = t > k ? t - k : 0; i <= std::min(t, m); ++i) s = add(s, mul(a[8 * i + l], b[8 * (t - i) + l]));
                        ok &= c[8 * t + l] == mul(s, inverse_r);
                    }
                expect(ok, "multiply_lanes", m, k);
            }
        }
    }
}

// Lanes: slot k, lane l holds polys[8k + l] (constant 1 past the end), lanes in any order of degree.
struct TestSlots {
    const std::vector<std::vector<u32>>* polys;

    std::size_t count() const { return (polys->size() + 7) / 8; }
    std::size_t lane_degree(std::size_t k, std::size_t l) const { return 8 * k + l < polys->size() ? (*polys)[8 * k + l].size() - 1 : 0; }
    u32 degree(std::size_t k) const {
        std::size_t d = 0;
        for (std::size_t l = 0; l < 8; ++l) d = std::max(d, lane_degree(k, l));
        return u32(d);
    }
    poly::LaneLayout::Node load(std::size_t k, u32* c) const {
        alignas(32) u32 degrees[8], leads[8];
        std::fill_n(c, 8 * (degree(k) + 1), 0);
        for (std::size_t l = 0; l < 8; ++l) {
            const std::vector<u32> one{1};
            const auto& f = 8 * k + l < polys->size() ? (*polys)[8 * k + l] : one;
            for (std::size_t j = 0; j < f.size(); ++j) c[8 * j + l] = to_m(f[j]);
            degrees[l] = u32(f.size() - 1), leads[l] = to_m(f.back());
        }
        return {poly::detail::load(degrees), poly::detail::load(leads)};
    }
};

struct TestItems {
    const std::vector<std::vector<u32>>* polys;  // Montgomery form

    std::size_t count() const { return polys->size(); }
    u32 degree(std::size_t k) const { return u32((*polys)[k].size() - 1); }
    poly::StandardLayout::Node load(std::size_t k, u32* c) const {
        const auto& f = (*polys)[k];
        std::copy(f.begin(), f.end(), c);
        return {u32(f.size() - 1), f.back()};
    }
};

// Each lane's product against the naive one (or at random points when long).
void check_lane_tree(const std::vector<std::vector<u32>>& polys) {
    static poly::Arena arena(poly::TreeTransform::words(18) + poly::Arena::footprint(std::size_t(1) << 22));
    static const poly::TreeTransform t(arena, 18);
    static const std::span<u32> scratch = arena.take(std::size_t(1) << 22);
    const TestSlots slots{&polys};
    std::size_t total = 0;
    for (std::size_t k = 0; k < slots.count(); ++k) total += slots.degree(k);
    if (8 * poly::LaneLayout::length(u32(total)) > (std::size_t(1) << 18) ||
        poly::ProductTree<poly::LaneLayout, TestSlots>::scratch_words(slots.count(), total) > scratch.size())
        std::abort();
    poly::ProductTree<poly::LaneLayout, TestSlots> tree(t, slots, scratch);
    const auto root = tree.root();
    alignas(32) u32 degrees[8];
    poly::detail::store(degrees, root.node.degree);
    for (std::size_t l = 0; l < 8; ++l) {
        std::vector<std::vector<u32>> lane;
        for (std::size_t i = l; i < polys.size(); i += 8) lane.push_back(polys[i]);
        std::size_t degree = 0;
        for (const auto& f : lane) degree += f.size() - 1;
        expect(degrees[l] == degree, "lane tree degree", l, degrees[l]);
        std::vector<u32> got(root.length + 1);
        for (std::size_t j = 0; j <= root.length; ++j) got[j] = from_m(root.coefficients[8 * j + l]);
        bool ok = true;
        for (std::size_t j = degree + 1; j <= root.length; ++j) ok &= got[j] == 0;
        if (degree <= 3000) {
            const auto want = naive_product(lane);
            ok &= std::equal(want.begin(), want.end(), got.begin());
        } else {
            for (int trial = 0; trial < 3; ++trial) {
                const u32 x = u32(rng() % P);
                u32 want = 1;
                for (const auto& f : lane) want = mul(want, evaluate(f, x));
                ok &= evaluate(got, x) == want;
            }
        }
        expect(ok, "lane tree product", polys.size(), l);
    }
}

// The product against the naive one; every node's kept transform against the forward of its product.
void check_standard_tree(const std::vector<std::vector<u32>>& polys, Fixture& fx) {
    static poly::Arena arena(poly::TreeTransform::words(18) + poly::Arena::footprint(std::size_t(1) << 22));
    static const poly::TreeTransform t(arena, 18);
    static const std::span<u32> scratch = arena.take(std::size_t(1) << 22);
    std::vector<std::vector<u32>> montgomery = polys;
    for (auto& f : montgomery)
        for (auto& x : f) x = to_m(x);
    const TestItems items{&montgomery};
    struct Keep {
        const std::vector<std::vector<u32>>* polys;
        const u32* roots;
        int* checked;
        void operator()(std::size_t lo, std::size_t hi, std::span<const u32> transform) const {
            const std::size_t n = transform.size();
            std::size_t degree = 0;
            for (std::size_t k = lo; k < hi; ++k) degree += (*polys)[k].size() - 1;
            if (n > 4096 || degree > 2000) return;
            std::vector<u32> node = naive_product({polys->begin() + std::ptrdiff_t(lo), polys->begin() + std::ptrdiff_t(hi)});
            for (std::size_t i = n; i < node.size(); ++i) node[i % n] = add(node[i % n], node[i]);  // mod x^n - 1
            node.resize(n);
            bool ok = true;
            for (std::size_t p = 0; p < n / 8; ++p) {
                const auto want = leaf(node, p, roots);
                for (std::size_t i = 0; i < 8; ++i) ok &= from_m(transform[8 * p + i]) == want[i];
            }
            expect(ok, "kept transform", lo, hi);
            ++*checked;
        }
    };
    int checked = 0;
    poly::ProductTree<poly::StandardLayout, TestItems, Keep> tree(t, items, scratch, Keep{&polys, fx.roots.data(), &checked});
    const auto root = tree.root();
    std::size_t degree = 0;
    for (const auto& f : polys) degree += f.size() - 1;
    expect(root.node.degree == degree, "standard tree degree", degree, root.node.degree);
    std::vector<u32> got(root.length + 1);
    for (std::size_t j = 0; j <= root.length; ++j) got[j] = from_m(root.coefficients[j]);
    bool ok = true;
    for (std::size_t j = degree + 1; j <= root.length; ++j) ok &= got[j] == 0;
    if (degree <= 3000) {
        const auto want = naive_product(polys);
        ok &= std::equal(want.begin(), want.end(), got.begin());
    } else {
        for (int trial = 0; trial < 3; ++trial) {
            const u32 x = u32(rng() % P);
            u32 want = 1;
            for (const auto& f : polys) want = mul(want, evaluate(f, x));
            ok &= evaluate(got, x) == want;
        }
    }
    expect(ok, "standard tree product", polys.size(), degree);
    expect(polys.size() == 1 || checked > 0, "kept transforms seen", polys.size());
}

// Degrees for a tree test: linear factors (often 2^k of them: wraps at every level), equal
// degrees, random small degrees, one large and many small, sizes near the base.
std::vector<std::size_t> tree_degrees(int shape, std::size_t total) {
    std::vector<std::size_t> d;
    switch (shape) {
        case 0: d.assign(total, 1); break;
        case 1: d.assign(std::max<std::size_t>(1, total / 7), 7); break;
        case 2:
            while (total) d.push_back(std::min<std::size_t>(total, 1 + rng() % 5)), total -= d.back();
            break;
        case 3:
            d.push_back(total * 3 / 4);
            for (std::size_t rest = total - total * 3 / 4; rest; --rest) d.push_back(1);
            break;
        default:
            while (total) d.push_back(std::min<std::size_t>(total, 1 + rng() % 70)), total -= d.back();
    }
    std::erase(d, 0);
    if (d.empty()) d.push_back(1);
    std::shuffle(d.begin(), d.end(), rng);
    if (shape == 4) std::sort(d.rbegin(), d.rend());
    return d;
}

void test_product_tree(Fixture& fx) {
    test_tree_transform(fx);
    test_multiply_lanes();
    const std::size_t sizes[] = {1, 2, 3, 7, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 1000, 1024, 3000};
    for (std::size_t total : sizes) {
        for (int shape = 0; shape < 5; ++shape) {
            const auto degrees = tree_degrees(shape, total);
            std::vector<std::vector<u32>> polys;
            for (std::size_t d : degrees) polys.push_back(random_degree(d, int(rng() % 3)));
            check_standard_tree(polys, fx);
            std::vector<std::vector<u32>> lanes;  // 8 lanes, each about this total
            for (int l = 0; l < 8; ++l)
                for (std::size_t d : tree_degrees(shape, total)) lanes.push_back(random_degree(d, int(rng() % 3)));
            std::stable_sort(lanes.begin(), lanes.end(), [](const auto& x, const auto& y) { return x.size() > y.size(); });
            check_lane_tree(lanes);
            std::shuffle(lanes.begin(), lanes.end(), rng);  // slots with mixed lane degrees
            check_lane_tree(lanes);
        }
    }
    for (std::size_t total : {std::size_t(1) << 12, (std::size_t(1) << 14) + 1, std::size_t(30000)}) {
        for (int shape : {0, 2, 3}) {
            std::vector<std::vector<u32>> polys;
            for (std::size_t d : tree_degrees(shape, total)) polys.push_back(random_degree(d, 0));
            check_standard_tree(polys, fx);
            std::vector<std::vector<u32>> lanes;
            for (int l = 0; l < 8; ++l)
                for (std::size_t d : tree_degrees(shape, total)) lanes.push_back(random_degree(d, 0));
            std::stable_sort(lanes.begin(), lanes.end(), [](const auto& x, const auto& y) { return x.size() > y.size(); });
            check_lane_tree(lanes);
        }
    }
    // lane_columns
    alignas(32) static u32 c[8 * 100];
    for (std::size_t i = 0; i < 8 * 100; ++i) c[i] = u32(i);
    static u32 out[8][104];
    u32 size[8] = {100, 0, 1, 7, 8, 9, 64, 99};
    u32* const columns[8] = {out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]};
    poly::lane_columns(c, 100, size, columns);
    bool ok = true;
    for (std::size_t l = 0; l < 8; ++l)
        for (std::size_t j = 0; j < size[l]; ++j) ok &= out[l][j] == 8 * j + l;
    expect(ok, "lane_columns");
}

// x_k = c s^k q^t(k) by x_(k+1) = x_k s q^k.
std::vector<u32> chirp_reference(u32 c, u32 s, u32 q, std::size_t n) {
    std::vector<u32> x(n);
    u32 term = c, ratio = s;
    for (std::size_t k = 0; k < n; ++k) x[k] = term, term = mul(term, ratio), ratio = mul(ratio, q);
    return x;
}

void test_chirp() {
    static constexpr u32 special[] = {0, 1, 2, P - 1};
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = trial < 100 ? std::size_t(trial) : trial % 10 == 0 ? (1 << 16) + pick(100) : pick(3000);
        const std::size_t offset = trial % 2;  // unaligned spans too
        const auto value = [trial](int which) { return trial % 5 == which ? special[pick(4)] : u32(rng() % P); };
        const u32 c = value(0), s = value(1), q = value(2);
        const auto want = chirp_reference(c, s, q, n);
        std::vector<u32> out(n + 2, 7);
        poly::chirp(c, s, q, std::span(out).subspan(offset, n));
        expect(std::equal(want.begin(), want.end(), out.begin() + offset) && out[offset + n] == 7, "chirp", n, trial);
        auto f = random_poly(n + 2, trial % 3);
        const auto g = f;
        poly::multiply_chirp(c, s, q, std::span(f).subspan(offset, n));
        bool ok = f[offset + n] == g[offset + n];
        for (std::size_t k = 0; k < n; ++k) ok &= f[offset + k] == mul(g[offset + k], want[k]);
        expect(ok, "multiply_chirp", n, trial);
    }
}

// Multipoint evaluation (evaluation.hpp).

// middle_lanes against scalar sums, every k <= 16 and k + count <= 32, and middle_lanes<D, D, B>
// for the blocks middle_level uses; coefficients random, 0 or near P - 1.
void test_middle_lanes() {
    alignas(32) static u32 w[8 * 32], q[8 * 17], out[8 * 32];
    const u32 inverse_r = power(to_m(1), P - 2);
    for (int kind = 0; kind < 3; ++kind) {
        for (std::size_t k = 0; k <= 16; ++k) {
            for (std::size_t count = 1; k + count <= 32; ++count) {
                const auto value = [kind] { return kind == 2 ? P - 1 - u32(rng() % 1024) : kind == 1 && rng() % 2 ? 0 : u32(rng() % P); };
                for (std::size_t i = 0; i < 8 * (k + count); ++i) w[i] = value();
                for (std::size_t i = 0; i < 8 * (k + 1); ++i) q[i] = value();
                namespace pd = poly::detail;
                const bool fixed = k == count && std::has_single_bit(k) && pd::middle_block(k);
                if (fixed) pd::with_degree(k, [&]<std::size_t D>() { if constexpr (pd::middle_block(D)) pd::middle_lanes<D, D, pd::middle_block(D)>(w, q, out); });
                else pd::middle_lanes(w, k, q, count, out);
                bool ok = true;
                for (std::size_t l = 0; l < 8; ++l)
                    for (std::size_t t = 0; t < count; ++t) {
                        u32 s = 0;
                        for (std::size_t j = 0; j <= k; ++j) s = add(s, mul(w[8 * (k + t - j) + l], q[8 * j + l]));
                        ok &= out[8 * t + l] == mul(s, inverse_r);
                    }
                expect(ok, fixed ? "middle_lanes<D, D, B>" : "middle_lanes", k, count);
            }
        }
    }
}

// Points of a kind: random, many zeros and repeats, small values, near P - 1.
std::vector<u32> random_points(std::size_t m, int kind) {
    std::vector<u32> a(m);
    for (auto& x : a) {
        const u64 r = rng();
        x = kind == 1 ? (r % 3 == 0 ? 0 : u32(r % 5)) : kind == 2 ? u32(r % 64) : kind == 3 ? P - 1 - u32(r % 16) : u32(r % P);
    }
    return a;
}

// The tree's values against Horner's rule (all points, or 42 when n m is large), and evaluate().
void check_evaluation(std::size_t n, std::size_t m, int kind) {
    auto f = random_poly(n, kind == 1 ? 1 : 0);
    if (f.back() == 0) f.back() = 1;
    const auto points = random_points(m, kind);
    std::vector<u32> got(m), direct(m);
    {
        poly::Arena arena(poly::detail::evaluate_tree_words(n, m));
        poly::detail::evaluate_tree(arena, f, points, got);
    }
    {
        poly::Arena arena(poly::evaluate_words(n, m));
        poly::evaluate(arena, f, points, direct);
    }
    std::vector<std::size_t> at;
    if (n * m <= (std::size_t(1) << 22)) {
        for (std::size_t i = 0; i < m; ++i) at.push_back(i);
    } else {
        at = {0, m - 1};
        for (int i = 0; i < 40; ++i) at.push_back(pick(m));
    }
    bool ok = true, same = true;
    for (std::size_t i : at) ok &= got[i] == evaluate(f, points[i]), same &= direct[i] == got[i];
    expect(ok, "evaluate_tree", n, m);
    expect(same, "evaluate", n, m);
    if (m % 8 == 0) {  // values aligned: written in place
        poly::Arena arena(poly::detail::evaluate_tree_words(n, m) + poly::Arena::footprint(m));
        const auto aligned = arena.take(m);
        poly::detail::evaluate_tree(arena, f, points, aligned);
        expect(equal(aligned, got), "evaluate_tree, aligned values", n, m);
    }
}

// The tree's product against the naive one.
void check_point_product(std::size_t m) {
    const auto points = random_points(m, int(rng() % 4));
    poly::Arena arena(poly::PointTree::words(m));
    poly::PointTree tree(arena, points);
    std::vector<std::vector<u32>> factors;
    for (u32 a : points) factors.push_back({1, a ? P - a : 0});
    auto want = naive_product(factors);
    want.resize(tree.size() + 1, 0);
    bool ok = tree.product().size() == want.size();
    for (std::size_t i = 0; ok && i < want.size(); ++i) ok &= from_m(tree.product()[i]) == want[i];
    expect(ok, "PointTree product", m);
}

// TreeTransform::inverse_upper undoes forward_upper (times c), in place and not, 8 .. 2^15 words.
void test_inverse_upper() {
    static poly::Arena arena(poly::TreeTransform::words(16) + 3 * poly::Arena::footprint(1 << 15));
    static const poly::TreeTransform t(arena, 16);
    static std::span<u32> buf[3] = {arena.take(1 << 15), arena.take(1 << 15), arena.take(1 << 15)};
    for (int lg = 3; lg <= 15; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (int kind = 0; kind < 3; ++kind) {
            const auto a = random_poly(n, kind);
            std::copy(a.begin(), a.end(), buf[0].begin());
            const auto upper = buf[1].first(n), out = buf[2].first(n);
            t.forward_upper(buf[0].first(n), 0, upper);
            const u32 c = kind == 0 ? 1 : kind == 1 ? P - 1 : u32(rng() % P);
            t.inverse_upper(upper, out, c);
            bool ok = true;
            for (std::size_t i = 0; i < n; ++i) ok &= out[i] == mul(c, a[i]);
            t.inverse_upper(upper, upper, c);
            ok &= std::equal(upper.begin(), upper.end(), out.begin());
            expect(ok, "TreeTransform inverse_upper", lg, kind);
        }
    }
}

void test_evaluation() {
    test_inverse_upper();
    test_middle_lanes();
    for (std::size_t m : {1, 2, 7, 8, 9, 64, 255, 256, 257, 300, 1000, 1024}) check_point_product(m);
    const std::size_t sizes[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 263, 264,
                                 511, 512, 513, 1000, 1024, 2047, 2048, 2049, 4095, 4096, 4097};
    for (std::size_t n : sizes)
        for (std::size_t m : sizes) check_evaluation(n, m, int(rng() % 4));
    const std::pair<std::size_t, std::size_t> large[] = {{1 << 17, 1 << 17}, {(1 << 17) - 20, 1 << 17}, {1 << 17, (1 << 17) - 1},
                                                         {100000, 77777}, {77777, 100000}, {1, 1 << 17}, {1 << 17, 1},
                                                         {5, 1 << 17}, {1 << 17, 9}, {65537, 65537}, {12345, 8193}};
    for (const auto& [n, m] : large)
        for (int kind = 0; kind < 4; ++kind) check_evaluation(n, m, kind);
    for (int trial = 0; trial < 40; ++trial) check_evaluation(1 + pick(20000), 1 + pick(20000), trial % 4);
}

// Interpolation (interpolation.hpp).

// TreeTransform's layout conversion and sums of products, 8 .. 2^13 words per polynomial:
// standard_to_lanes of the standard transforms of 8 polynomials against their lanes transform;
// leaf_product_sums and pointwise_product_sums against the sums of the products (Montgomery form).
void test_tree_layouts() {
    constexpr std::size_t kWords = (1 << 16) + 128;  // 8 columns of 2^13 words, 16 apart
    static poly::Arena arena(poly::TreeTransform::words(16) + 8 * poly::Arena::footprint(kWords));
    static const poly::TreeTransform t(arena, 16);
    static std::span<u32> buf[8] = {arena.take(kWords), arena.take(kWords), arena.take(kWords), arena.take(kWords),
                                    arena.take(kWords), arena.take(kWords), arena.take(kWords), arena.take(kWords)};
    for (int lg = 3; lg <= 13; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        for (int kind = 0; kind < 3; ++kind) {
            const auto lanes = buf[0].first(8 * n), standard = buf[1].first(8 * n + 16 * 7), out = buf[3].first(8 * n);
            const std::size_t stride = n + 16 * (kind == 1);
            for (std::size_t l = 0; l < 8; ++l) {
                const auto f = random_poly(n, kind);
                for (std::size_t i = 0; i < n; ++i) lanes[8 * i + l] = f[i];
                const auto single = standard.subspan(l * stride, n);
                std::copy(f.begin(), f.end(), single.begin());
                t.forward(single, 0, single);
            }
            t.forward(lanes, 0, lanes);
            t.standard_to_lanes(standard.data(), stride, n, out.data());
            expect(std::equal(out.begin(), out.end(), lanes.begin()), "standard_to_lanes", lg, kind);

            std::vector<u32> x[4];
            std::span<u32> tx[4];
            for (int k = 0; k < 4; ++k) {
                x[k] = random_poly(n, (kind + k) % 3);
                tx[k] = buf[4 + k].first(n);
                for (std::size_t i = 0; i < n; ++i) tx[k][i] = to_m(x[k][i]);
            }
            poly::TreeTransform::pointwise_product_sums(tx[0].data(), tx[1].data(), tx[2].data(), tx[3].data(), out.first(n));
            bool ok = true;
            for (std::size_t i = 0; i < n; ++i) ok &= out[i] == to_m(add(mul(x[0][i], x[1][i]), mul(x[2][i], x[3][i])));
            expect(ok, "pointwise_product_sums", lg, kind);
            for (int k = 0; k < 4; ++k) t.forward(tx[k], 0, tx[k]);
            t.leaf_product_sums(tx[0].data(), tx[1].data(), tx[2].data(), tx[3].data(), out.first(n));
            t.inverse(out.first(n), out.first(n));
            ok = true;
            for (std::size_t i : {std::size_t(0), n - 1, pick(n), pick(n)})
                ok &= from_m(out[i]) == add(cyclic_coefficient(x[0], x[1], i), cyclic_coefficient(x[2], x[3], i));
            expect(ok, "leaf_product_sums", lg, kind);
        }
    }
}

// cross_lanes against scalar sums: every n, k <= 16 (loops), and the unrolled and blocked versions
// for n = k = 1 .. 16; coefficients random, 0 or near P - 1.
void test_cross_lanes() {
    alignas(32) static u32 a[8 * 16], b[8 * 16], c[8 * 17], d[8 * 17], out[8 * 32];
    const u32 inverse_r = power(to_m(1), P - 2);
    namespace pd = poly::detail;
    for (int kind = 0; kind < 3; ++kind) {
        const auto value = [kind] { return kind == 2 ? P - 1 - u32(rng() % 1024) : kind == 1 && rng() % 2 ? 0 : u32(rng() % P); };
        for (auto* v : {a, b}) std::generate_n(v, 8 * 16, value);
        for (auto* v : {c, d}) std::generate_n(v, 8 * 17, value);
        const auto check = [&](std::size_t n, std::size_t k, const char* what) {
            bool ok = true;
            for (std::size_t l = 0; l < 8; ++l)
                for (std::size_t t = 0; t < n + k; ++t) {
                    u32 s = 0;
                    for (std::size_t i = 0; i < n; ++i)
                        if (t >= i && t - i <= k) s = add(s, mul(a[8 * i + l], d[8 * (t - i) + l]));
                    for (std::size_t i = 0; i < k; ++i)
                        if (t >= i && t - i <= n) s = add(s, mul(b[8 * i + l], c[8 * (t - i) + l]));
                    ok &= out[8 * t + l] == mul(s, inverse_r);
                }
            expect(ok, what, n, k);
        };
        for (std::size_t n = 1; n <= 16; ++n)
            for (std::size_t k = 1; k <= 16; ++k) {
                pd::cross_lanes(a, n, b, k, c, d, out);
                check(n, k, "cross_lanes");
            }
        pd::cross_lanes<1>(a, b, c, d, out), check(1, 1, "cross_lanes<D>");
        pd::cross_lanes<2>(a, b, c, d, out), check(2, 2, "cross_lanes<D>");
        pd::cross_lanes<4>(a, b, c, d, out), check(4, 4, "cross_lanes<D>");
        pd::cross_lanes<8>(a, b, c, d, out), check(8, 8, "cross_lanes<D>");
        pd::cross_lanes<5, 3>(a, b, c, d, out), check(5, 5, "cross_lanes<D, B>");
        pd::cross_lanes<16, 2>(a, b, c, d, out), check(16, 16, "cross_lanes<D, B>");
    }
}

// m distinct points of a kind: random, 0 .. m - 1 shuffled, near P - 1, 0 and random.
std::vector<u32> distinct_points(std::size_t m, int kind) {
    std::vector<u32> a;
    if (kind == 1 || kind == 2) {
        for (std::size_t i = 0; i < m; ++i) a.push_back(kind == 1 ? u32(i) : P - 1 - u32(i));
        std::shuffle(a.begin(), a.end(), rng);
        return a;
    }
    while (a.size() < m) {
        for (std::size_t i = a.size(); i < m; ++i) a.push_back(u32(rng() % P));
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
    std::shuffle(a.begin(), a.end(), rng);
    if (kind == 3) a[pick(m)] = 0, std::sort(a.begin(), a.end()), a.erase(std::unique(a.begin(), a.end()), a.end());
    return a;
}

// interpolate() against the points' values by Horner's rule (all of them, or 42 when m is large).
void check_interpolation(std::size_t m, int kind) {
    const auto points = distinct_points(m, kind);
    m = points.size();
    const auto values = random_poly(m, kind % 3);
    std::vector<u32> c(m);
    poly::Arena arena(poly::interpolate_words(m));
    poly::interpolate(arena, points, values, c);
    std::vector<std::size_t> at;
    if (m <= 2048) {
        for (std::size_t i = 0; i < m; ++i) at.push_back(i);
    } else {
        at = {0, m - 1};
        for (int i = 0; i < 40; ++i) at.push_back(pick(m));
    }
    bool ok = true;
    for (std::size_t i : at) ok &= evaluate(c, points[i]) == values[i];
    expect(ok, "interpolate", m, kind);
}

void test_interpolation() {
    test_tree_layouts();
    test_cross_lanes();
    for (std::size_t m = 1; m <= 80; ++m) check_interpolation(m, int(m % 4));
    for (std::size_t m : {255, 256, 257, 263, 264, 265, 511, 512, 513, 520, 1000, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097})
        for (int kind = 0; kind < 4; ++kind) check_interpolation(m, kind);
    for (std::size_t m : {65537, 100000, (1 << 17) - 1, 1 << 17})
        for (int kind = 0; kind < 4; ++kind) check_interpolation(m, kind);
    for (int trial = 0; trial < 30; ++trial) check_interpolation(1 + pick(20000), trial % 4);
}

// Newton basis (newton.hpp).

// Points of a kind, repeats allowed: random, mostly 0, mostly P - 1, all one value.
std::vector<u32> newton_points(std::size_t n, int kind) {
    if (kind == 3) return std::vector<u32>(n, u32(rng() % P));
    return random_poly(n, kind);
}

// to_newton() against repeated synthetic division (n <= 3000: c_k = g(p_k), g <- g / (x - p_k)),
// else f(x) = sum_k c_k prod_(i < k) (x - p_i) at 8 random x.
void check_newton(std::size_t n, int kind) {
    const auto f = random_poly(n, kind % 3), points = newton_points(n, kind);
    std::vector<u32> c(n, 7);
    poly::Arena arena(poly::to_newton_words(n));
    poly::to_newton(arena, f, points, c);
    if (n <= 3000) {
        std::vector<u32> g = f, want(n);
        for (std::size_t k = 0; k < n; ++k) {
            u32 carry = 0;  // g[j] <- g[j] + p_k g[j + 1] from the top: g[0] = g(p_k), g[1 ..) the quotient
            for (std::size_t j = n; j-- > k;) g[j] = carry = add(g[j], mul(carry, points[k]));
            want[k] = g[k];
        }
        expect(c == want, "to_newton", n, kind);
        return;
    }
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        const u32 x = u32(rng() % P);
        u32 sum = 0, basis = 1;
        for (std::size_t k = 0; k < n; ++k) sum = add(sum, mul(c[k], basis)), basis = mul(basis, sub(x, points[k]));
        ok &= sum == evaluate(f, x);
    }
    expect(ok, "to_newton identity", n, kind);
}

void test_newton() {
    for (std::size_t n = 1; n <= 100; ++n) check_newton(n, int(n % 4));
    for (std::size_t n : {255, 256, 257, 263, 264, 265, 511, 512, 513, 520, 1000, 1024, 1025, 2047, 2048, 2049, 2056, 2999})
        for (int kind = 0; kind < 4; ++kind) check_newton(n, kind);
    for (std::size_t n : {4095, 4096, 4097, 8200, 65534, 65535, 65536, 65537, 65538, 80000, 85192, 85200, 100000, (1 << 17) - 1, 1 << 17})
        for (int kind = 0; kind < 4; ++kind) check_newton(n, kind);
    for (int trial = 0; trial < 30; ++trial) check_newton(1 + pick(trial < 20 ? 3000 : 40000), trial % 4);
}

// Factorials and product chains (factorials.hpp).

// factorial and factorials against running products for every n < kFactorialLimit at random,
// table boundaries and the limit; invert against products; scan_chunk's bounds.
void test_factorial_values() {
    static std::vector<u32> fact(poly::kFactorialLimit);
    fact[0] = 1;
    for (u32 i = 1; i < poly::kFactorialLimit; ++i) fact[i] = mul(fact[i - 1], i);
    for (u32 n : {0u, 1u, 2u, 1023u, 1024u, 1025u, 2047u, 2048u, poly::kFactorialLimit - 1})
        expect(poly::factorial(n) == fact[n], "factorial", n);
    for (int trial = 0; trial < 300; ++trial) {
        poly::Lanes n;
        for (u32& x : n) x = trial % 3 == 0 ? u32(pick(3000)) : trial % 3 == 1 ? u32(1024 * pick(1025)) : u32(pick(poly::kFactorialLimit));
        if (trial % 7 == 0) n[pick(32)] = poly::kFactorialLimit - 1;
        const poly::Lanes f = poly::factorials(n);
        bool ok = true;
        for (int s = 0; s < 32; ++s) ok &= f[s] == fact[n[s]];
        expect(ok, "factorials", trial);
        poly::Lanes g = f;
        poly::invert(g);
        ok = true;
        for (int s = 0; s < 32; ++s) ok &= mul(f[s], g[s]) == 1;
        expect(ok, "invert lanes", trial);
    }
    const auto check_invert = [](auto x) {
        for (u32& v : x) v = rng() % 4 ? 1 + u32(rng() % (P - 1)) : rng() % 2 ? 1 : P - 1;
        auto y = x;
        poly::invert(y);
        bool ok = true;
        for (std::size_t i = 0; i < x.size(); ++i) ok &= mul(x[i], y[i]) == 1;
        expect(ok, "invert", x.size());
    };
    check_invert(std::array<u32, 0>{});
    check_invert(std::array<u32, 1>{});
    check_invert(std::array<u32, 2>{});
    check_invert(std::array<u32, 33>{});
    check_invert(std::array<u32, 1000>{});
    for (std::size_t n = 1; n < 1 << 21; n = n < 5000 ? n + 1 : n * 3 / 2 + pick(100)) {
        const std::size_t c = poly::detail::scan_chunk(n);
        const bool minimal = c == 16 || 32 * (c - 32) < n;  // the previous candidate is c - 32
        expect(c % 16 == 0 && c / 16 % 2 == 1 && 32 * c >= n && 32 * c <= n + 1023 && minimal, "scan_chunk", n);
    }
}

// Chains through scan, forward and reversed, against scalar products: lane s multiplies by
// (base_s + j step) / 2^32 at step j. Starts below 2P, bases and steps random or near 0 and P.
void test_chains() {
    const u32 r_inverse = power(to_m(1), P - 2);
    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t steps = 8 * (1 + pick(trial < 20 ? 4 : 64));
        poly::Lanes x, base;
        for (int s = 0; s < 32; ++s) {
            x[s] = trial % 5 == 0 ? 2 * P - 1 - u32(pick(4)) : u32(rng() % (2 * P));
            base[s] = trial % 4 == 0 ? P - 1 - u32(pick(4)) : u32(rng() % P);
        }
        const u32 step = trial % 3 == 0 ? P - to_m(1) : trial % 3 == 1 ? to_m(1) : u32(rng() % P);
        poly::detail::Chain<> up(x, base, step);
        poly::detail::Chain<true> down(x, base, step);
        std::vector<u32> forward(32 * steps), reversed(32 * steps);
        poly::detail::scan(steps, [&](std::size_t j, int s, poly::detail::Vec a, poly::detail::Vec b) {
            poly::detail::store_unaligned(forward.data() + s * steps + j, a);
            poly::detail::store_unaligned(reversed.data() + s * steps + j, b);
        }, up, down);
        bool ok = true;
        for (int s = 0; s < 32; ++s) {
            u32 term = x[s] % P;
            for (std::size_t j = 0; j < steps; ++j) {
                const std::size_t block = j / 8 * 8, t = j % 8;
                ok &= forward[s * steps + j] < 2 * P && forward[s * steps + j] % P == term;
                ok &= reversed[s * steps + block + 7 - t] % P == term;
                term = mul(mul(term, u32((base[s] + u64(j) * step) % P)), r_inverse);
            }
        }
        expect(ok, "chains", trial, steps);
    }
}

void test_factorials() {
    test_factorial_values();
    test_chains();
}

// q and r of f by g by long division.
std::pair<std::vector<u32>, std::vector<u32>> long_divide(std::vector<u32> f, const std::vector<u32>& g) {
    const std::size_t d = g.size() - 1;
    std::vector<u32> q(f.size() > d ? f.size() - d : 0);
    const u32 lead = power(g[d], P - 2);
    for (std::size_t i = q.size(); i-- > 0;) {
        q[i] = mul(f[i + d], lead);
        for (std::size_t t = 0; t <= d; ++t) f[i + t] = sub(f[i + t], mul(q[i], g[t]));
    }
    f.resize(std::min(f.size(), d));
    return {q, f};
}

// divide() with plan (or the default plan if null) against long division (n m <= 2^22), else
// f = q g + r at 8 random points.
void check_division(const std::vector<u32>& f, const std::vector<u32>& g, const poly::detail::DivisionPlan* plan) {
    const std::size_t n = f.size(), m = g.size();
    std::vector<u32> q(poly::quotient_size(n, m), 7), r(poly::remainder_size(n, m), 7);
    poly::Arena arena(plan ? plan->words() : poly::divide_words(n, m));
    if (plan) {
        poly::detail::divide(arena, *plan, f, g, q, r);
    } else {
        poly::divide(arena, f, g, q, r);
    }
    const std::size_t s = plan ? plan->quotient.s : 0;
    if (n * m <= (std::size_t(1) << 22)) {
        const auto [eq, er] = long_divide(f, g);
        expect(q == eq, "division quotient", n * 1000000 + m, s);
        expect(r == er, "division remainder", n * 1000000 + m, s);
        return;
    }
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        const u32 x = u32(rng() % P);
        ok &= evaluate(f, x) == add(mul(evaluate(q, x), evaluate(g, x)), evaluate(r, x));
    }
    expect(ok, "division identity", n * 1000000 + m, s);
}

// Every block size (with at most kDivisionTerms windows) and every remainder method that applies.
void check_division_plans(std::size_t n, std::size_t m, int kind) {
    const auto f = random_poly(n, kind), g = random_degree(m - 1, kind);
    check_division(f, g, nullptr);
    if (n < m || m == 1) return;
    const std::size_t k = n - m + 1;
    for (std::size_t s = 32; s < 2 * k || s == 32; s *= 2) {
        using poly::detail::Remainder;
        for (auto how : {Remainder::kDirect, Remainder::kWrap, Remainder::kSplit, Remainder::kTail}) {
            const poly::detail::DivisionPlan plan(n, m, s, how);
            constexpr double kInfinity = std::numeric_limits<double>::infinity();
            if (plan.quotient.ns != kInfinity && plan.remainder.ns != kInfinity) check_division(f, g, &plan);
        }
    }
}

void test_division() {
    for (std::size_t n = 1; n <= 40; ++n)
        for (std::size_t m = 1; m <= 40; ++m) check_division(random_poly(n, int(n % 3)), random_degree(m - 1, int(m % 3)), nullptr);
    // Blocks: tails and windows (up to 8 terms at s = 32: k = 288, d = 300); folds of q and g
    // (d = 64, 128: L = d); short quotients.
    const std::size_t sizes[][2] = {{100, 37},  {200, 65},    {200, 64},    {300, 129},  {588, 301},  {600, 33},
                                    {1000, 500}, {1000, 10},   {2000, 1999}, {2000, 1990}, {3000, 2950}, {4096, 1025},
                                    {5000, 2},   {5000, 4500}, {5000, 4936}, {3000, 1000}, {2049, 1024}, {1500, 1500}};
    for (const auto& [n, m] : sizes)
        for (int kind = 0; kind < 3; ++kind) check_division_plans(n, m, kind);
    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t n = 1 + pick(3000), m = 1 + pick(n + 10);
        check_division_plans(n, m, trial % 3);
    }
    const std::size_t large[][2] = {{(1 << 18) + 5, (1 << 14) + 3}, {300000, 150000}, {1 << 18, (1 << 17) + 1},
                                    {200000, 199990},                 {500000, 53336},  {500000, 499999},
                                    {300000, 10},                     {200000, 1 << 16},
                                    {500000, 277012},                 {500000, 389813}};
    for (const auto& [n, m] : large) check_division(random_poly(n), random_degree(m - 1, 0), nullptr);
    for (std::size_t s : {1 << 12, 1 << 15}) {
        const poly::detail::DivisionPlan plan(200000, 30000, s, poly::detail::Remainder::kWrap);
        check_division(random_poly(200000), random_degree(29999, 0), &plan);
    }
}

using Poly = std::vector<u32>;

void trim(Poly& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

long degree(const Poly& a) { return long(a.size()) - 1; }

Poly poly_product(const Poly& a, const Poly& b) {
    if (a.empty() || b.empty()) return {};
    Poly c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    trim(c);
    return c;
}

// a + sign b, sign = 1 or P - 1.
Poly poly_sum(Poly a, const Poly& b, u32 sign = 1) {
    if (a.size() < b.size()) a.resize(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) a[i] = add(a[i], mul(sign, b[i]));
    trim(a);
    return a;
}

std::pair<Poly, Poly> poly_divmod(const Poly& a, const Poly& b) {
    auto [q, r] = long_divide(a, b);
    trim(q);
    trim(r);
    return {q, r};
}

// The extended Euclidean algorithm on (a, b) while deg r1 >= target: s0 a + t0 b = r0, s1 a + t1 b = r1.
struct EuclidRows {
    Poly s0, t0, r0, s1, t1, r1;
};

EuclidRows euclid_rows(const Poly& a, const Poly& b, long target) {
    EuclidRows w{{1}, {}, a, {}, {1}, b};
    while (!w.r1.empty() && degree(w.r1) >= target) {
        const auto [q, r] = poly_divmod(w.r0, w.r1);
        Poly s = poly_sum(w.s0, poly_product(q, w.s1), P - 1), t = poly_sum(w.t0, poly_product(q, w.t1), P - 1);
        w.s0 = w.s1, w.t0 = w.t1, w.r0 = w.r1;
        w.s1 = s, w.t1 = t, w.r1 = r;
    }
    return w;
}

// x = c y for a nonzero constant c (both trimmed).
bool proportional(const Poly& x, const Poly& y) {
    if (x.size() != y.size()) return false;
    if (x.empty()) return true;
    const u32 c = mul(x.back(), power(y.back(), P - 2));
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i] != mul(c, y[i])) return false;
    return true;
}

// (a, b) with deg a = d whose remainder sequence has quotients of degree 1 .. q_max, both times a
// random common factor of degree common.
std::pair<Poly, Poly> euclid_pair(long d, long q_max, long common) {
    Poly a, b{1};
    while (degree(b) < d) {
        Poly next = poly_sum(a, poly_product(b, random_degree(1 + pick(std::size_t(q_max)), 0)));
        a = b, b = next;
    }
    const Poly h = random_degree(std::size_t(common), 0);
    return {poly_product(b, h), poly_product(a, h)};
}

// The jump of k on (a, b) against the Euclidean algorithm: rows and remainders proportional to
// its rows, progress n - deg r_h. direct: Euclid's algorithm for jumps of at most that many degrees.
void check_jump(const Poly& a, const Poly& b, long k, long direct) {
    const long n = degree(a), m = degree(b);
    poly::Arena arena(poly::detail::HalfGcd::words(std::size_t(n)));
    poly::detail::HalfGcd gcd(arena, std::size_t(n), direct);
    poly::detail::Matrix r = gcd.matrix(std::size_t(k), 0);
    gcd.jump(a.data(), n, b.data(), m, k, r);
    const EuclidRows w = euclid_rows(a, b, n - k);
    Poly e[4];
    for (int i = 0; i < 4; ++i) {
        e[i].assign(r.entry[i], r.entry[i] + r.size[i]);
        trim(e[i]);
    }
    bool ok = proportional(e[0], w.s0) && proportional(e[1], w.t0) && proportional(e[2], w.s1) && proportional(e[3], w.t1);
    ok &= long(r.progress) == n - degree(w.r0);
    ok &= proportional(poly_sum(poly_product(e[0], a), poly_product(e[1], b)), w.r0);
    ok &= proportional(poly_sum(poly_product(e[2], a), poly_product(e[3], b)), w.r1);
    for (int i = 0; i < 4; ++i) ok &= r.at0[i] == (e[i].empty() ? 0 : e[i][0]);
    expect(ok, "gcd jump", u64(n) * 1000000 + u64(m), u64(k) * 1000 + u64(direct));
}

// inverse_mod against the extended Euclidean algorithm on (g, f mod g).
void check_inverse_mod(const Poly& f, const Poly& g) {
    poly::Arena arena(poly::inverse_mod_words(f.size(), g.size()));
    Poly h(g.size(), 7);
    const std::ptrdiff_t t = poly::inverse_mod(arena, f, g, h);
    Poly want;
    long want_t = 0;
    if (degree(g) > 0) {
        Poly reduced = f;
        trim(reduced);
        const EuclidRows w = euclid_rows(g, poly_divmod(reduced, g).second, 0);
        if (degree(w.r0) != 0) {
            want_t = -1;
        } else {
            const u32 c = power(w.r0[0], P - 2);
            for (u32 x : w.t0) want.push_back(mul(x, c));
            want_t = long(want.size());
        }
    }
    const bool ok = t == want_t && (t <= 0 || Poly(h.begin(), h.begin() + t) == want);
    expect(ok, "inverse_mod", f.size() * 1000000 + g.size(), u64(t + 1));
}

// inverse_mod of random f, g of n and m coefficients: (f h - 1) mod g = 0.
void check_inverse_mod_large(std::size_t n, std::size_t m) {
    const Poly f = random_degree(n - 1, 0), g = random_degree(m - 1, 0);
    poly::Arena arena(poly::inverse_mod_words(n, m));
    Poly h(m);
    const std::ptrdiff_t t = poly::inverse_mod(arena, f, g, h);
    if (t <= 0) return expect(false, "inverse_mod large", n, m);
    ntt::Convolution product(n, std::size_t(t));
    std::copy(f.begin(), f.end(), product.a());
    std::copy_n(h.begin(), t, product.b());
    const u32* fh = product.multiply();
    Poly e(fh, fh + n + std::size_t(t) - 1);
    e[0] = sub(e[0], 1);
    Poly q(poly::quotient_size(e.size(), m)), r(poly::remainder_size(e.size(), m));
    poly::Arena divide_arena(poly::divide_words(e.size(), m));
    poly::divide(divide_arena, e, g, q, r);
    expect(std::all_of(r.begin(), r.end(), [](u32 x) { return x == 0; }), "inverse_mod large", n, m);
}

void test_gcd() {
    const long directs[] = {1, 2, 3, 8, 32, poly::detail::HalfGcd::kDirect, 100};
    for (int trial = 0; trial < 150; ++trial) {
        const long n = 1 + long(pick(300));
        Poly a, b;
        if (trial % 4 == 3) {
            std::tie(a, b) = euclid_pair(n, 1 + long(pick(30)), long(pick(5)));
        } else {
            a = random_degree(std::size_t(n), trial % 4);
            b = random_degree(trial % 3 ? pick(std::size_t(n)) : std::size_t(n - 1), trial % 4);
        }
        const long k = 1 + long(pick(std::size_t(degree(a))));
        check_jump(a, b, k, directs[trial % 7]);
        check_jump(a, b, degree(a), directs[(trial + 3) % 7]);
    }
    for (std::size_t n = 1; n <= 12; ++n)
        for (std::size_t m = 1; m <= 12; ++m) check_inverse_mod(random_poly(n, int(n % 3)), random_degree(m - 1, int(m % 3)));
    for (int trial = 0; trial < 120; ++trial) {
        const std::size_t n = 1 + pick(500), m = 1 + pick(500);
        if (trial % 5 == 4) {  // abnormal sequences, some with a common factor
            const auto [g, f] = euclid_pair(long(std::max(n, m)), 1 + long(pick(30)), trial % 3 ? 0 : 1 + long(pick(4)));
            check_inverse_mod(f, g);
            continue;
        }
        Poly f = random_degree(n - 1, trial % 3), g = random_degree(m - 1, trial % 3);
        if (trial % 5 == 3) {  // a common factor, or g dividing f
            const Poly h = random_degree(pick(5), 0);
            f = poly_product(f, h), g = poly_product(g, h);
            if (trial % 2) f = poly_product(g, random_degree(pick(5), 0));
        }
        check_inverse_mod(f, g);
    }
    for (std::size_t d : {64, 65, 127, 128, 129, 1000, 1024, 1025}) {
        check_inverse_mod(random_degree(d + pick(3), 0), random_degree(d, 0));
        check_inverse_mod(random_degree(pick(d + 1), 0), random_degree(d, 0));
        check_inverse_mod(Poly{5}, random_degree(d, 0));
    }
    const std::size_t large[][2] = {{50000, 50000}, {20000, 50000}, {50000, 3000}, {32769, 32769}, {1000, 40000}};
    for (const auto& [n, m] : large) check_inverse_mod_large(n, m);
}

int main() {
    static Fixture fx;
    test_leaf_kernels(fx);
    test_factors(fx);
    test_column_kernels(fx);
    test_transforms(fx);
    test_inverse(fx);
    test_inverse_2d();
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
    test_product_tree(fx);
    test_chirp();
    test_evaluation();
    test_interpolation();
    test_newton();
    test_factorials();
    test_division();
    test_gcd();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
}
