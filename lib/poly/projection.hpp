// Power projection modulo 998244353: a[k] = [x^(n-1)] g^k for k < n and g[0] = 0, by Kinoshita
// and Li's algorithm: composition.hpp's levels, run forward together with the numerator.
// x86-64 with AVX2. Design: lib/poly/notes.md (Power projection).
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::projection_log(n)) + poly::projection_scratch(n) + ...);
//   poly::Transform t(arena, poly::projection_log(n));
//   poly::power_projection(t, g, a, arena.take(poly::projection_scratch(n)));  // a.size() == n
//
// With m = 2^T >= n: sum_k a[k] y^k = [x^(m-1)] P_0 / Q_0 for P_0 = x^(m-n) and Q_0 = 1 - y g(x).
// Level s has L = m / 2^s coefficients in x, Q_s of degree Y = 2^s in y and P_s of degree below
// Y, and [x^(L-1)] P_s / Q_s is the same series. With A = P_s(x) Q_s(-x): P_(s+1)(x^2) x = the
// odd part of A mod x^L and Q_(s+1)(x^2) = Q_s(x) Q_s(-x) mod x^L (the denominator is even, and
// L - 1 is odd). At s = T, L = 1 and Q_T(0, y) = 1: a = P_T. Layouts and transforms as
// composition.hpp: Kronecker x = z, y = z^(2L) at length 4m; P_s fills rows 0 .. Y - 1, z below 2m.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/composition.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// a[k] = [x^(n-1)] g^k for k < n = a.size() <= kComposeBase, from the powers of g.
inline void projection_direct(std::span<const std::uint32_t> g, std::span<std::uint32_t> a) {
    const std::size_t n = a.size();
    std::uint32_t power[kComposeBase] = {1}, next[kComposeBase];  // g^k mod x^n
    for (std::size_t k = 0; k < n; ++k) {
        a[k] = power[n - 1];
        for (std::size_t i = 0; i < n; ++i) {
            std::uint64_t c = 0;  // fewer than kComposeBase terms < P
            for (std::size_t j = 1; j <= i && j < g.size(); ++j) c += std::uint64_t(g[j]) * power[i - j] % kP;
            next[i] = std::uint32_t(c % kP);
        }
        std::copy_n(next, n, power);
    }
}

// Leaf products in transposed form, one leaf per lane, over one parity of lanes: each 64-bit
// lane multiplies the low 32 bits (the even lanes as given, the odd lanes after a shift).
inline Vec product(Vec x, Vec y) { return _mm256_mul_epu32(x, y); }
inline Vec twice(Vec x) { return _mm256_add_epi64(x, x); }
inline Vec plus(Vec x, Vec y) { return _mm256_add_epi64(x, y); }
inline Vec minus(Vec x, Vec y) { return _mm256_sub_epi64(x, y); }

// c = Q mod (z^8 - t) = e(u) + z o(u), u = z^2 (coefficient k in c[k], canonical), and
// tc[k] = t c[k] for k >= 4 (canonical): sum = e^2 - u o^2 mod (u^4 - t), the leaf of Q(z) Q(-z),
// as 64-bit sums. Each has 4 negative terms, offset by 4 P^2, and stays below 8 P^2.
[[gnu::always_inline]] inline void graeffe_sums(const Vec (&c)[8], const Vec (&tc)[8], Vec (&sum)[4]) {
    const Vec offset = _mm256_set1_epi64x(static_cast<long long>(4 * std::uint64_t(kP) * kP));
    const auto x = [&c](int i, int j) { return product(c[i], c[j]); };
    const auto t = [&c, &tc](int i, int j) { return product(c[i], tc[j]); };
    sum[0] = minus(minus(plus(plus(x(0, 0), twice(t(2, 6))), plus(t(4, 4), offset)), twice(t(1, 7))), twice(t(3, 5)));
    sum[1] = minus(minus(plus(plus(twice(x(0, 2)), twice(t(4, 6))), offset), x(1, 1)), plus(twice(t(3, 7)), t(5, 5)));
    sum[2] = minus(minus(plus(plus(twice(x(0, 4)), x(2, 2)), plus(t(6, 6), offset)), twice(x(1, 3))), twice(t(5, 7)));
    sum[3] = minus(minus(plus(plus(twice(x(0, 6)), twice(x(2, 4))), offset), twice(x(1, 5))), plus(x(3, 3), t(7, 7)));
}

// a = P mod (z^8 - t), c and tc as for graeffe_sums (tc[k] for k >= 2): sum = the odd part of
// a(z) c(-z) mod (z^8 - t), as a series in u mod (u^4 - t): sum_k adds (-1)^j a_i c_j over
// i + j = 2k + 1 and (-1)^j a_i tc_j over i + j = 2k + 9. 8 terms, 4 of them negative (offset by
// 4 P^2), below 8 P^2.
[[gnu::always_inline]] inline void odd_product_sums(const Vec (&a)[8], const Vec (&c)[8], const Vec (&tc)[8],
                                                     Vec (&sum)[4]) {
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
        Vec s = _mm256_set1_epi64x(static_cast<long long>(4 * std::uint64_t(kP) * kP));
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) {
            const int j = 2 * k + 1 - i, l = 2 * k + 9 - i;
            if (j >= 0 && j < 8) s = j % 2 ? minus(s, product(a[i], c[j])) : plus(s, product(a[i], c[j]));
            if (l < 8) s = l % 2 ? minus(s, product(a[i], tc[l])) : plus(s, product(a[i], tc[l]));
        }
        sum[k] = s;
    }
}

// The sums over the even lanes, then over the odd lanes, reduced: times 2^-32, in [0, 2P).
[[gnu::always_inline]] inline void graeffe_leaf(const Vec (&c)[8], const Vec (&tc)[8], Vec (&out)[4]) {
    Vec even[4], odd[4], c2[8], tc2[8];
    graeffe_sums(c, tc, even);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) c2[i] = _mm256_srli_epi64(c[i], 32), tc2[i] = _mm256_srli_epi64(tc[i], 32);
    graeffe_sums(c2, tc2, odd);
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) out[k] = low(reduce_wide({even[k], odd[k]}));
}

[[gnu::always_inline]] inline void odd_product_leaf(const Vec (&a)[8], const Vec (&c)[8], const Vec (&tc)[8],
                                                     Vec (&out)[4]) {
    Vec even[4], odd[4], a2[8], c2[8], tc2[8];
    odd_product_sums(a, c, tc, even);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i)
        a2[i] = _mm256_srli_epi64(a[i], 32), c2[i] = _mm256_srli_epi64(c[i], 32), tc2[i] = _mm256_srli_epi64(tc[i], 32);
    odd_product_sums(a2, c2, tc2, odd);
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) out[k] = low(reduce_wide({even[k], odd[k]}));
}

// The bottom of the forward transform of Q_s (length 4m): forward butterflies, then per pair of
// leaves 2k, 2k + 1 (moduli z^8 -+ s, s = r[k]) the leaves of V = Q_s(x) Q_s(-x) and of
// W = the odd part of P(x) Q_s(-x), from p, P's transform (canonical leaves): mod (u^4 -+ s),
// u = z^2, A and B by graeffe_sums and odd_product_sums; leaf k of the transform of length 2m is
// (A + B) + u^4 (A - B) / s, which is twice the CRT of A and B. With the 2^-32 of the products,
// V and W come out times 2^-31, in [0, 2P); their inverse transforms undo it. Leaf k goes to
// v + 8k and w + 8k. Pairs go 8 at a time, one per lane, after their leaves are read, so v may be
// Q's array and w may be p: each writes below the leaves read so far.
struct ProjectionBottom {
    static constexpr bool kForward = true, kInverse = false;
    const Tables* tables;
    const std::uint32_t* p;
    std::uint32_t *v, *w;

    void operator()(std::uint32_t* a, std::size_t count, std::size_t first) const {
        ForwardBottom{tables->roots}(a, count, first);
        for (std::size_t i = 0; i < count; i += 4) pairs(a + 32 * i, 2 * (first + i));
    }

    // 8 pairs from Q's leaves at q, the first pair k (a multiple of 8). Flattened: GCC kept the
    // transposes as calls, with every vector spilled around them.
    [[gnu::flatten]] void pairs(const std::uint32_t* q, std::size_t k) const {
        Factor s(0), inverse(0);  // r[k + lane] and its inverse
        s.w = load(tables->roots + slot(k)), s.q = load(tables->roots + slot(k) + 8);
        inverse.w = load(tables->inverse_roots + slot(k)), inverse.q = load(tables->inverse_roots + slot(k) + 8);
        Vec va[4], vb[4], wa[4], wb[4];
        leaves<false>(q, p + 16 * k, s, va, wa);
        leaves<true>(q + 8, p + 16 * k + 8, s, vb, wb);
        combine(va, vb, inverse, v + 8 * k);
        combine(wa, wb, inverse, w + 8 * k);
    }

    // V and W mod (u^4 - t) for t = s (or -s if kNegative) of the 8 leaves at q and p, stride 16,
    // times 2^-32, in [0, 2P): the sums over the even lanes, then over the odd lanes.
    template <bool kNegative>
    [[gnu::noinline, gnu::flatten]] static void leaves(const std::uint32_t* q, const std::uint32_t* p, const Factor& s,
                                                       Vec (&v)[4], Vec (&w)[4]) {
        Vec c[8], a[8], tc[8];
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) c[i] = load(q + 16 * i), a[i] = load(p + 16 * i);
        transpose(c), transpose(a);
        tc[0] = tc[1] = _mm256_setzero_si256();  // not read
#pragma GCC unroll 6
        for (int i = 2; i < 8; ++i) {
            const Vec x = reduce(times_lanes(c[i], s), kP);
            tc[i] = kNegative ? negate(x) : x;
        }
        graeffe_leaf(c, tc, v);
        odd_product_leaf(a, c, tc, w);
    }

    // Leaves k .. k + 7 at out from x mod (u^4 - s) and y mod (u^4 + s): (x + y) + u^4 (x - y) / s.
    static void combine(const Vec (&x)[4], const Vec (&y)[4], const Factor& inverse, std::uint32_t* out) {
        Vec r[8];
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) r[i] = low(add(x[i], y[i])), r[i + 4] = times_lanes(diff(x[i], y[i]), inverse);
        transpose(r);
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) store(out + 8 * i, r[i]);
    }
};

// Pruned transforms for the Kronecker layouts, in vectors of 8 coefficients. In the forward
// transform of a level, vector bit b of the index (x = L, the padding bit) is zero in the input;
// the levels above b (y) act on each column separately, so they skip the columns with bit b set,
// and the level with bit b has half its inputs. In the inverse transforms of V and W only the
// columns with bit b clear (x < L/2) are kept: the y levels compute those, the level with bit b
// half its outputs. Levels below b are full (Recursion), and so are small groups (kForwardFull,
// kInverseFull): the bottoms take at least 16 vectors, and narrow columns gain little.

// Butterflies j < h of a radix-4 group (inputs at j + t h) on the columns with bit b of j clear.
inline void forward_columns(std::uint32_t* a, std::size_t h, std::size_t b, const Group& w) {
    const std::size_t width = std::size_t(1) << b;
    for (std::size_t c = 0; c < h; c += 2 * width)
        for (std::uint32_t* x = a + 8 * c; x < a + 8 * (c + width); x += 8) {
            Vec f[4] = {load(x), load(x + 8 * h), load(x + 16 * h), load(x + 24 * h)};
            forward_h1(f, w);
            store(x, f[0]), store(x + 8 * h, f[1]), store(x + 16 * h, f[2]), store(x + 24 * h, f[3]);
        }
}

// The same with inverse butterflies.
inline void inverse_columns(std::uint32_t* a, std::size_t h, std::size_t b, const Group& w) {
    const std::size_t width = std::size_t(1) << b;
    for (std::size_t c = 0; c < h; c += 2 * width)
        for (std::uint32_t* x = a + 8 * c; x < a + 8 * (c + width); x += 8) {
            Vec f[4] = {load(x), load(x + 8 * h), load(x + 16 * h), load(x + 24 * h)};
            inverse_h1(f, w);
            store(x, f[0]), store(x + 8 * h, f[1]), store(x + 16 * h, f[2]), store(x + 24 * h, f[3]);
        }
}

// forward_h1 with the inputs at 2h and 3h zero (high = true) or at h and 3h (high = false).
inline void forward_half(std::uint32_t* a, std::size_t h, bool high, const Group& w) {
    for (std::uint32_t* x = a; x < a + 8 * h; x += 8) {
        const Vec u = low(load(x));
        if (high) {
            const Vec b = load(x + 8 * h), yb = times(b, w.y), zb = times(b, w.z);
            store(x, add(u, yb)), store(x + 8 * h, diff(u, yb));
            store(x + 16 * h, add(u, zb)), store(x + 24 * h, diff(u, zb));
        } else {
            const Vec xc = times(load(x + 16 * h), w.x), sum = low(add(u, xc)), difference = low_difference(u, xc);
            store(x, sum), store(x + 8 * h, sum), store(x + 16 * h, difference), store(x + 24 * h, difference);
        }
    }
}

// inverse_h1 computing only the outputs at 0 and h (high = true: 2h and 3h unused) or at 0 and 2h.
inline void inverse_half(std::uint32_t* a, std::size_t h, bool high, const Group& w) {
    for (std::uint32_t* x = a; x < a + 8 * h; x += 8) {
        const Vec f0 = load(x), f1 = load(x + 8 * h), f2 = load(x + 16 * h), f3 = load(x + 24 * h);
        const Vec ab = low(add(f0, f1)), cd = low(add(f2, f3));
        store(x, low(add(ab, cd)));
        if (high) store(x + 8 * h, low(add(times(diff(f0, f1), w.y), times(diff(f2, f3), w.z))));
        else store(x + 16 * h, times(diff(ab, cd), w.x));
    }
}

// The pruned forward and inverse below the top level: group k of nv vectors at a.
template <class Bottom>
class Pruned {
public:
    Pruned(const Tables& tables, const Bottom& bottom, std::size_t b)
        : recursion_(tables.roots, tables.inverse_roots, bottom), tables_(tables), b_(b) {}

    void forward(std::uint32_t* a, std::size_t nv, std::size_t k) const {
        const std::size_t h = nv / 4, low = std::size_t(std::countr_zero(h));
        if (nv <= kForwardFull || low + 1 < b_) return recursion_.visit(a, nv, k);
        const Group w(tables_.roots, k);
        if (low > b_) {
            forward_columns(a, h, b_, w);
            for (std::size_t t = 0; t < 4; ++t) forward(a + 8 * t * h, h, 4 * k + t);
            return;
        }
        forward_half(a, h, low + 1 == b_, w);
        for (std::size_t t = 0; t < 4; ++t) recursion_.visit(a + 8 * t * h, h, 4 * k + t);
    }

    void inverse(std::uint32_t* a, std::size_t nv, std::size_t k) const {
        const std::size_t h = nv / 4, low = std::size_t(std::countr_zero(h));
        if (nv <= kInverseFull || low + 1 < b_) return recursion_.visit(a, nv, k);
        const Group w(tables_.inverse_roots, k);
        if (low > b_) {
            for (std::size_t t = 0; t < 4; ++t) inverse(a + 8 * t * h, h, 4 * k + t);
            return inverse_columns(a, h, b_, w);
        }
        for (std::size_t t = 0; t < 4; ++t) recursion_.visit(a + 8 * t * h, h, 4 * k + t);
        inverse_half(a, h, low + 1 == b_, w);
    }

private:
    static constexpr std::size_t kForwardFull = 64, kInverseFull = 16;  // vectors

    Recursion<Bottom> recursion_;
    const Tables& tables_;
    std::size_t b_;
};

// The forward transform of length a.size() of a[0, size) in place (the rest zero and not read),
// with vector bit b of the index zero in the input; b lies below the top level's bits. The top
// level writes zeros to the skipped columns, which the levels above b then leave alone.
template <class Bottom>
void forward_pruned(std::span<std::uint32_t> a, std::size_t size, std::size_t b, const Tables& tables,
                    const Bottom& bottom) {
    const Pruned<Bottom> pruned(tables, bottom, b);
    const Source in(a.data(), size, 0);
    const std::size_t nv = a.size() / 8, width = std::size_t(1) << b;
    auto* v = reinterpret_cast<Vec*>(a.data());
    const Vec p = broadcast(kP), zero = _mm256_setzero_si256();
    if (std::countr_zero(nv) % 2 == 0) {  // radix-4 identity group, as forward_top4
        const std::size_t h = nv / 4;
        const Factor z(tables.roots[1], tables.roots[9]);
        for (std::size_t c = 0; c < h; c += 2 * width) {
            for (std::size_t j = c; j < c + width; ++j) {
                const Vec x0 = in(j), x1 = in(j + h), x2 = in(j + 2 * h), x3 = in(j + 3 * h);
                const Vec ac = add(x0, x2), amc = _mm256_sub_epi32(add(x0, p), x2);
                const Vec bd = add(x1, x3), zbmd = times(_mm256_sub_epi32(add(x1, p), x3), z);
                v[j] = add(ac, bd), v[j + h] = diff(ac, bd);
                v[j + 2 * h] = add(amc, zbmd), v[j + 3 * h] = diff(amc, zbmd);
            }
            for (std::size_t j = c + width; j < c + 2 * width; ++j)
                v[j] = v[j + h] = v[j + 2 * h] = v[j + 3 * h] = zero;
        }
        for (std::size_t t = 0; t < 4; ++t) pruned.forward(a.data() + 8 * t * h, h, t);
    } else {  // radix 2, as forward_top2
        const std::size_t h = nv / 2;
        for (std::size_t c = 0; c < h; c += 2 * width) {
            for (std::size_t j = c; j < c + width; ++j) {
                const Vec x0 = in(j), x1 = in(j + h);
                v[j] = add(x0, x1), v[j + h] = _mm256_sub_epi32(add(x0, p), x1);
            }
            for (std::size_t j = c + width; j < c + 2 * width; ++j) v[j] = v[j + h] = zero;
        }
        pruned.forward(a.data(), h, 0);
        pruned.forward(a.data() + 8 * h, h, 1);
    }
}

// The inverse transform of length a.size() in place, times scale, canonical on the columns with
// vector bit b clear (the others are left unspecified); b lies below the top level's bits.
template <class Bottom>
void inverse_pruned(std::span<std::uint32_t> a, std::size_t b, const Tables& tables, const Bottom& bottom,
                    std::uint32_t scale) {
    const Pruned<Bottom> pruned(tables, bottom, b);
    const std::size_t nv = a.size() / 8, width = std::size_t(1) << b;
    auto* v = reinterpret_cast<Vec*>(a.data());
    const Factor s(scale);
    const auto scaled = [&s](Vec x) { return reduce(times(x, s), kP); };
    if (std::countr_zero(nv) % 2 == 0) {  // radix-4 identity group, as inverse_top4
        const std::size_t h = nv / 4;
        for (std::size_t t = 0; t < 4; ++t) pruned.inverse(a.data() + 8 * t * h, h, t);
        const Factor z(tables.inverse_roots[1], tables.inverse_roots[9]);
        for (std::size_t c = 0; c < h; c += 2 * width)
            for (std::size_t j = c; j < c + width; ++j) {
                const Vec x0 = v[j], x1 = v[j + h], x2 = v[j + 2 * h], x3 = v[j + 3 * h];
                const Vec ab = low(add(x0, x1)), cd = low(add(x2, x3));
                const Vec amb = low(diff(x0, x1)), cmd = times(diff(x2, x3), z);
                v[j] = scaled(add(ab, cd)), v[j + h] = scaled(add(amb, cmd));
                v[j + 2 * h] = scaled(diff(ab, cd)), v[j + 3 * h] = scaled(diff(amb, cmd));
            }
    } else {  // radix 2, as inverse_top2
        const std::size_t h = nv / 2;
        pruned.inverse(a.data(), h, 0);
        pruned.inverse(a.data() + 8 * h, h, 1);
        for (std::size_t c = 0; c < h; c += 2 * width)
            for (std::size_t j = c; j < c + width; ++j) {
                const Vec x0 = v[j], x1 = v[j + h];
                v[j] = scaled(add(x0, x1)), v[j + h] = scaled(diff(x0, x1));
            }
    }
}

// Level 0, one-dimensional, into q and p (length 4m each): Q_0 = 1 - y g and P_0 = x^e, e = m - n.
// From the transform of g at 2m, LevelBottom gives g(x) g(-x) = v(u). At stride m:
// Q_1 = 1 - 2y ge(u) + y^2 v(u) with g = ge(x^2) + x go(x^2), and P_1 = [e odd] u^((e-1)/2) - y W(u)
// with W the odd part of x^e g(-x): W[i] = (-1)^j g_j for j = 2i + 1 - e; both mod u^(m/2).
inline void projection_first_level(const Transform& t, const Tables& tables, std::span<const std::uint32_t> g,
                                   std::size_t n, std::span<std::uint32_t> q, std::span<std::uint32_t> p) {
    const std::size_t m = q.size() / 4, h = m / 2, e = m - n, size = std::min(n, g.size());
    const auto at = [&](std::size_t i) { return i > 0 && i < size ? g[i] : 0; };
    for (std::size_t i = 0; i < size; ++i) q[i] = at(i);
    forward_with(q.first(2 * m), size, tables, LevelBottom{&tables, q.data()});
    t.inverse(q.first(m));
    std::copy(q.begin(), q.begin() + std::ptrdiff_t(h), q.begin() + std::ptrdiff_t(2 * m));
    std::fill(q.begin() + std::ptrdiff_t(2 * m + h), q.begin() + std::ptrdiff_t(3 * m), 0);
    for (std::size_t j = 0; j < h; ++j) q[m + j] = minus_twice(at(2 * j));
    std::fill(q.begin() + std::ptrdiff_t(m + h), q.begin() + std::ptrdiff_t(2 * m), 0);
    std::fill(q.begin(), q.begin() + std::ptrdiff_t(m), 0);
    q[0] = 1;
    std::fill(p.begin(), p.begin() + std::ptrdiff_t(2 * m), 0);
    if (e % 2) p[(e - 1) / 2] = 1;
    for (std::size_t i = e / 2; i < h; ++i) {  // -W
        const std::size_t j = 2 * i + 1 - e;
        const std::uint32_t x = at(j);
        p[m + i] = j % 2 || !x ? x : kP - x;
    }
}

// Levels T - 2 and T - 1, one-dimensional in y, from v and w (level T - 3's V and W at stride 8,
// m/4 rows, v wrapped: row 0 holds 1 + row m/4): Q_(T-2) = 1 + x q1 + x^2 q2 + x^3 q3 and
// P_(T-2) = p0 + x p1 + x^2 p2 + x^3 p3, deg q_k <= m/4, deg p_k < m/4. With c_k = (-1)^k q_k,
// a = [x^3] P_(T-2) / Q_(T-2) = r3 + r1 (c1^2 - 2 c2) for r1 = p1 + p0 c1 and
// r3 = p3 + p2 c1 + p1 c2 + p0 c3 (products of length m/2), the last product of length m.
// c1^2 has no y^0 term, so its y^(m/2) wraps onto y^0. carve: 9 spans of m/2, 4 of m/4, 2 of m.
inline void projection_last_levels(const Transform& t, std::span<const std::uint32_t> v,
                                   std::span<const std::uint32_t> w, std::span<std::uint32_t> a, Carve& carve) {
    const std::size_t m = v.size() / 2, h = m / 2, rows = m / 4;
    std::span<std::uint32_t> c[4], pc[4], pt[3];  // c[k], k >= 1: transforms; pc: coefficients; pt: transforms of pc
    for (std::size_t k = 1; k < 4; ++k) {
        c[k] = carve.take(h);
        std::fill(c[k].begin(), c[k].end(), 0);
        for (std::size_t i = 1; i <= rows; ++i) {
            const std::uint32_t x = v[8 * (i % rows) + k];
            c[k][i] = k % 2 && x ? kP - x : x;
        }
    }
    for (std::size_t k = 0; k < 4; ++k) {
        pc[k] = carve.take(rows);
        for (std::size_t i = 0; i < rows; ++i) pc[k][i] = w[8 * i + k];
    }
    const std::span<std::uint32_t> square = carve.take(h), r1 = carve.take(h), r3 = carve.take(h);
    const std::span<std::uint32_t> q = carve.take(m), product = carve.take(m);
    std::fill(q.begin(), q.end(), 0);  // c1^2 - 2 c2
    for (std::size_t i = 1; i <= rows; ++i) q[i] = minus_twice(c[2][i]);
    for (std::size_t k = 1; k < 4; ++k) t.forward(c[k]);
    for (std::size_t k = 0; k < 3; ++k) {
        pt[k] = carve.take(h);
        t.forward(pc[k], 0, pt[k]);
    }
    t.inverse_product(c[1], c[1], square);
    for (std::size_t i = 1; i <= h; ++i) {
        const std::uint32_t x = q[i] + square[i % h];
        q[i] = x >= kP ? x - kP : x;
    }
    t.inverse_product(pt[0], c[1], r1);
    const Transform::Pair pairs[3] = {{pt[2], c[1]}, {pt[1], c[2]}, {pt[0], c[3]}};
    t.inverse_product_sum(pairs, r3);
    for (std::size_t i = 0; i < rows; ++i) {
        const std::uint32_t x = r1[i] + pc[1][i], y = r3[i] + pc[3][i];
        r1[i] = x >= kP ? x - kP : x, r3[i] = y >= kP ? y - kP : y;
    }
    t.forward(q);
    t.cyclic_product(r1, 0, product, q);
    for (std::size_t k = 0; k < a.size(); ++k) {
        const std::uint32_t x = (k < h ? r3[k] : 0) + product[k];
        a[k] = x >= kP ? x - kP : x;
    }
}

// Zero x >= half in each of rows rows at stride 2 half.
inline void truncate_rows(std::span<std::uint32_t> a, std::size_t half, std::size_t rows) {
    for (std::size_t i = 0; i < rows; ++i)
        std::fill(a.begin() + std::ptrdiff_t((2 * i + 1) * half), a.begin() + std::ptrdiff_t((2 * i + 2) * half), 0);
}

}  // namespace detail

// Transform length power_projection() uses for n outputs: the Transform needs lg_max >= this.
inline int projection_log(std::size_t n) { return compose_log(n); }

// Scratch words for power_projection() of n outputs.
inline std::size_t projection_scratch(std::size_t n) {
    using detail::Carve;
    if (n <= detail::kComposeBase) return 0;
    const std::size_t m = std::bit_ceil(std::max<std::size_t>(n, 128));
    return Arena::footprint(detail::Tables::words(m)) + 2 * Carve::words(4 * m) + 9 * Carve::words(m / 2) +
           4 * Carve::words(m / 4) + 2 * Carve::words(m);
}

// a[k] = [x^(n-1)] g^k for k < n = a.size() >= 1, g[0] = 0 (if g is not empty); coefficients
// past g.size() are zero. scratch: projection_scratch(n) words, 32-byte aligned (from an Arena).
// t: lg_max >= projection_log(n). a must not overlap g.
//
// Levels as above: 0 (one-dimensional in x, projection_first_level), 1 .. T - 3 (Kronecker
// layout: transforms of P_s and Q_s at 4m, the second with ProjectionBottom; inverses of V and W
// at 2m), T - 2 and T - 1 (one-dimensional in y, projection_last_levels).
inline void power_projection(const Transform& t, std::span<const std::uint32_t> g, std::span<std::uint32_t> a,
                             std::span<std::uint32_t> scratch) {
    using namespace detail;
    using ntt::detail::multiply_mod, ntt::detail::power;
    const std::size_t n = a.size();
    if (n <= kComposeBase) return projection_direct(g, a);
    const std::size_t m = std::bit_ceil(std::max<std::size_t>(n, 128));
    const int lg = std::countr_zero(m);
    const Tables tables(scratch.first(Tables::words(m)), m);
    Carve carve(scratch.subspan(Arena::footprint(Tables::words(m))));
    const std::span<std::uint32_t> q = carve.take(4 * m), p = carve.take(4 * m);
    projection_first_level(t, tables, g, n, q, p);
    // Inverses at 2m: undo the factor 2m / 8 and ProjectionBottom's 2^-31.
    const std::uint32_t scale = multiply_mod(power(std::uint32_t(m / 4), kP - 2), std::uint32_t((1ULL << 31) % kP));
    for (int s = 1; s + 2 < lg; ++s) {
        const std::size_t stride = 2 * (m >> s), rows = std::size_t(1) << s;  // Q_s: rows 0 .. Y; P_s: 0 .. Y - 1
        const std::size_t pad = std::size_t(std::countr_zero(stride / 16));  // vector bit of x = L
        forward_pruned(p, rows * stride, pad, tables, ForwardBottom{tables.roots});
        forward_pruned(q, (rows + 1) * stride, pad, tables, ProjectionBottom{&tables, p.data(), q.data(), p.data()});
        if (s + 3 < lg) {  // only x < L/2 is kept
            inverse_pruned(q.first(2 * m), pad - 1, tables, InverseBottom{tables.inverse_roots, q.data()}, scale);
            inverse_pruned(p.first(2 * m), pad - 1, tables, InverseBottom{tables.inverse_roots, p.data()}, scale);
            next_level(q, stride / 2, 2 * rows);
            truncate_rows(p, stride / 4, 2 * rows);
        } else {  // level T - 3: columns of V and W at stride 8 for the last levels
            inverse_with(q.first(2 * m), tables, InverseBottom{tables.inverse_roots, q.data()}, scale, Half::kBoth);
            inverse_with(p.first(2 * m), tables, InverseBottom{tables.inverse_roots, p.data()}, scale, Half::kBoth);
        }
    }
    projection_last_levels(t, q.first(2 * m), p.first(2 * m), a, carve);
}

}  // namespace poly
