// Composition of power series modulo 998244353: h = f(g) mod x^n for g[0] = 0, by Kinoshita and
// Li's algorithm (the transpose of power projection). x86-64 with AVX2. Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::compose_log(n)) + poly::compose_scratch(n) + ...);
//   poly::Transform t(arena, poly::compose_log(n));
//   poly::compose(t, f, g, h, arena.take(poly::compose_scratch(n)));  // h.size() == n
//
// The levels (shared with power projection): m = 2^T >= n, Q_0(x, y) = 1 - y g(x), and
// Q_(s+1)(x^2, y) = Q_s(x, y) Q_s(-x, y) mod x^(m / 2^s). Q_s has L = m / 2^s coefficients in x
// and degree Y = 2^s in y; Q_s(x, 0) = Q_s(0, y) = 1. A level 0 < s < T - 2 is stored as the
// transform of length 4m of Q_s by Kronecker substitution x = z, y = z^(2L): x below L, rows
// 0 .. Y of 2Y. A product mod (z^4m - 1) then carries nothing from x into y while the x degree
// stays below 2L, and wraps y mod y^(2Y). Levels 0, T - 2 and T - 1 are one-dimensional (Levels).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// Up to this many coefficients, Horner's rule.
inline constexpr std::size_t kComposeBase = 32;

// h = f(g) mod x^n by Horner's rule, n = h.size() <= kComposeBase. h = h g + f_i from the top
// coefficient down, so h[k] reads only h[0, k) (g[0] = 0).
inline void compose_direct(std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                           std::span<std::uint32_t> h) {
    const std::size_t n = h.size();
    std::fill(h.begin(), h.end(), 0);
    for (std::size_t i = std::min(f.size(), n); i-- > 0;) {
        for (std::size_t k = n; k-- > 1;) {
            std::uint64_t sum = 0;  // fewer than kComposeBase terms < P
            for (std::size_t j = 1; j <= k && j < g.size(); ++j) sum += std::uint64_t(g[j]) * h[k - j] % kP;
            h[k] = std::uint32_t(sum % kP);
        }
        h[0] = f[i];
    }
}

// -2x mod P for x < P.
constexpr std::uint32_t minus_twice(std::uint32_t x) {
    const std::uint32_t y = x ? 2 * kP - 2 * x : 0;  // < 2P
    return y >= kP ? y - kP : y;
}

// -x mod P in [0, P) for x < P.
inline Vec negate(Vec x) {
    return _mm256_min_epu32(_mm256_sub_epi32(broadcast(kP), x), _mm256_sub_epi32(_mm256_setzero_si256(), x));
}

// Lane j of r[i] <-> lane i of r[j].
inline void transpose(Vec (&r)[8]) {
    Vec t[8], u[8];
#pragma GCC unroll 4
    for (int i = 0; i < 8; i += 2) t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]), t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
#pragma GCC unroll 2
    for (int i = 0; i < 8; i += 4)
#pragma GCC unroll 2
        for (int j = 0; j < 2; ++j)
            u[i + 2 * j] = _mm256_unpacklo_epi64(t[i + j], t[i + j + 2]), u[i + 2 * j + 1] = _mm256_unpackhi_epi64(t[i + j], t[i + j + 2]);
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i)
        r[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20), r[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
}

// 64-bit lanes: the products of the even and of the odd 32-bit lanes.
struct Wide {
    Vec even, odd;
};

// A vector with its odd lanes also moved to the even ones, the form wide() multiplies.
struct Lanes {
    Vec even, odd;
};

inline Lanes lanes(Vec x) { return {x, _mm256_srli_epi64(x, 32)}; }
inline Wide wide(Lanes x, Lanes y) { return {_mm256_mul_epu32(x.even, y.even), _mm256_mul_epu32(x.odd, y.odd)}; }
inline Wide operator+(Wide x, Wide y) { return {_mm256_add_epi64(x.even, y.even), _mm256_add_epi64(x.odd, y.odd)}; }
inline Wide operator-(Wide x, Wide y) { return {_mm256_sub_epi64(x.even, y.even), _mm256_sub_epi64(x.odd, y.odd)}; }

// c P^2 in each 64-bit lane.
inline Wide squares_of_p(std::uint64_t c) {
    const Vec x = _mm256_set1_epi64x(static_cast<long long>(c * kP * kP));
    return {x, x};
}

// x / 2^32 mod P in [0, x / 2^32 + P) for 64-bit lanes x < 14 P^2 (Montgomery).
inline Vec reduce_wide(Wide x) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
    const Vec even = _mm256_add_epi64(x.even, _mm256_mul_epu32(_mm256_mul_epu32(x.even, ni), p));
    const Vec odd = _mm256_add_epi64(x.odd, _mm256_mul_epu32(_mm256_mul_epu32(x.odd, ni), p));
    return _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
}

// One leaf a mod (z^8 - s) per lane, c[k] = its coefficient k (canonical), s_m = s 2^32 mod P:
// out = a(z) a(-z) mod (z^8 - s) = e(u)^2 - u o(u)^2 mod (u^4 - s), u = z^2, a = e(u) + z o(u),
// times 2^-32, in [0, 2P). Every sum stays below 8 P^2. Always inlined, as CompositionBottom::leaf:
// GCC made both calls (compose 1% slower).
[[gnu::always_inline]] inline void leaf_graeffe(const Vec (&c)[8], Vec s_m, Vec (&out)[4]) {
    const Lanes e0 = lanes(c[0]), o0 = lanes(c[1]), e1 = lanes(c[2]), o1 = lanes(c[3]);
    const Lanes e2 = lanes(c[4]), o2 = lanes(c[5]), e3 = lanes(c[6]), o3 = lanes(c[7]);
    const Wide e01 = wide(e0, e1), e02 = wide(e0, e2), e03 = wide(e0, e3), e12 = wide(e1, e2), e13 = wide(e1, e3);
    const Wide e23 = wide(e2, e3), o01 = wide(o0, o1), o02 = wide(o0, o2), o03 = wide(o0, o3), o12 = wide(o1, o2);
    const Wide o13 = wide(o1, o3), o23 = wide(o2, o3);
    // out_k = main_k + s wrapped_k
    const Wide main[4] = {wide(e0, e0), e01 + e01 + squares_of_p(1) - wide(o0, o0),
                          e02 + e02 + wide(e1, e1) + squares_of_p(2) - o01 - o01,
                          e03 + e03 + e12 + e12 + squares_of_p(3) - o02 - o02 - wide(o1, o1)};
    const Wide wrapped[4] = {e13 + e13 + wide(e2, e2) + squares_of_p(4) - o03 - o03 - o12 - o12,
                             e23 + e23 + squares_of_p(3) - o13 - o13 - wide(o2, o2),
                             wide(e3, e3) + squares_of_p(2) - o23 - o23, squares_of_p(1) - wide(o3, o3)};
    const Lanes s = lanes(s_m);
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) out[k] = low(reduce_wide(main[k] + wide(lanes(canonical(reduce_wide(wrapped[k]))), s)));
}

// Twiddle tables for transforms of length 4m and, per pair p of their leaves (leaves 2p, 2p + 1:
// moduli z^8 -+ s, s = r[p]), s 2^32 and s^-1 2^63 mod P.
struct Tables {
    std::uint32_t *roots, *inverse_roots, *s_m, *hi_graeffe;

    static std::size_t words(std::size_t m) {
        const int lg = std::countr_zero(m) + 2;
        return 2 * Arena::footprint(ntt::detail::table_words(lg)) + 2 * Arena::footprint(m / 4);
    }

    Tables(std::span<std::uint32_t> memory, std::size_t m) {
        const std::size_t table = Arena::footprint(ntt::detail::table_words(std::countr_zero(m) + 2)), count = m / 4;
        roots = memory.data(), inverse_roots = roots + table, s_m = inverse_roots + table;
        hi_graeffe = s_m + Arena::footprint(count);
        ntt::detail::build_table(roots, count, ntt::detail::kRoots[0]);
        ntt::detail::build_table(inverse_roots, count, ntt::detail::kRoots[1]);
        const auto power_of_two = [](int e) {  // 2^e mod P
            std::uint64_t x = 1;
            for (int i = 0; i < e; ++i) x = x * 2 % kP;
            return std::uint32_t(x);
        };
        const Vec c64 = broadcast(power_of_two(64)), c95 = broadcast(power_of_two(95));  // montgomery() takes 2^-32
        for (std::size_t p = 0; p < count; p += 8) {
            store(s_m + p, reduce(montgomery(load(roots + slot(p)), c64), kP));
            store(hi_graeffe + p, reduce(montgomery(load(inverse_roots + slot(p)), c95), kP));
        }
    }
};

// The bottom of the forward transform of Q_s (length 4m): forward butterflies (canonical leaves,
// kept as the level), then, unless graeffe is null, the transform of length 2m of
// V = Q_s(x) Q_s(-x) at stride L (2Y rows, wrapped). For leaves a = Q mod (z^8 - s),
// b = Q mod (z^8 + s): V mod (u^4 -+ s) = A, B = a(z) a(-z), b(z) b(-z) (leaf_graeffe, u = z^2), and
// leaf p of V's transform is V mod (u^8 - w_p) = (A + B) / 2 + u^4 (A - B) / (2s), since s^2 = w_p.
// Pairs go 8 at a time (two groups); tiles hold at least 16 vectors (4m >= 256).
struct LevelBottom {
    static constexpr bool kForward = true, kInverse = false;
    const Tables* tables;
    std::uint32_t* graeffe;

    void operator()(std::uint32_t* a, std::size_t count, std::size_t first) const {
        ForwardBottom{tables->roots}(a, count, first);
        if (graeffe)
            for (std::size_t i = 0; i < count; i += 4) pairs(a + 32 * i, 2 * (first + i));
    }

    // 8 pairs from leaves at q, the first pair p.
    void pairs(const std::uint32_t* q, std::size_t p) const {
        Vec a[8], b[8];
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) a[i] = load(q + 16 * i), b[i] = load(q + 16 * i + 8);
        transpose(a), transpose(b);
        const Vec s_m = load(tables->s_m + p);
        Vec va[4], vb[4], v[8];
        leaf_graeffe(a, s_m, va);
        leaf_graeffe(b, negate(s_m), vb);
        // va, vb carry 2^-32: (A + B) / 2 = (va + vb) 2^31, (A - B) / (2s) = (va - vb) 2^63 / (s 2^32).
        const Vec half = broadcast(std::uint32_t((std::uint64_t(1) << 63) % kP)), hi = load(tables->hi_graeffe + p);
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            v[k] = reduce(montgomery(add(va[k], vb[k]), half), kP);
            v[k + 4] = reduce(montgomery(diff(va[k], vb[k]), hi), kP);
        }
        transpose(v);
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) store(graeffe + 8 * (p + i), v[i]);
    }
};

// The bottom of the inverse transform of R = P(z^2) Q_s(-z) (length 4m), from f, the transform of
// length 2m of P, and q, that of Q_s. For leaves 2p, 2p + 1 (moduli z^8 - w, w = +-s, s = r[p]),
// with u = z^2: P(z^2) mod (z^8 - w) = p(u) = lo + w hi, where leaf p of f is
// P mod (u^8 - s^2) = lo + u^4 hi; and with c = Q_s mod (z^8 - w) = ce(u) + z co(u),
// R mod (z^8 - w) = p(u) ce(u) - z p(u) co(u) mod (u^4 - w). Pairs go 8 at a time, one per lane
// (coefficients transposed); tiles hold at least 16 vectors (4m >= 256). Products times 2^-32
// (undone by the scale), then inverse butterflies.
struct CompositionBottom {
    static constexpr bool kForward = false, kInverse = true;
    const Tables* tables;
    const std::uint32_t *f, *q;

    // One leaf per lane: c[k] = coefficient k of Q_s mod (z^8 - w), p[i] = p_i and wp[i] = w p_i
    // (canonical). out[k] = coefficient k of R mod (z^8 - w), in [0, 2P). Each sum has 4 products.
    [[gnu::always_inline]] static void leaf(const Vec (&p)[4], const Vec (&wp)[4], const Vec (&c)[8], Vec (&out)[8]) {
        Lanes pl[4], wl[4], cl[8];
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) pl[i] = lanes(p[i]), wl[i] = lanes(wp[i]);
#pragma GCC unroll 8
        for (int k = 0; k < 8; ++k) cl[k] = lanes(c[k]);
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            Wide even{}, odd{};  // u^i c_(k-i): p_i c_(k-i) for i <= k, w p_i c_(k-i+4) for i > k
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i) {
                const Lanes& x = i <= k ? pl[i] : wl[i];
                const int j = (k - i + 4) % 4;
                even = even + wide(x, cl[2 * j]);
                odd = odd + wide(x, cl[2 * j + 1]);
            }
            out[2 * k] = reduce_wide(even);
            out[2 * k + 1] = reduce(_mm256_sub_epi32(broadcast(2 * kP), reduce_wide(odd)), 2 * kP);
        }
    }

    // Leaves 2p .. 2p + 15 of R into r.
    void products(std::size_t p, Vec (&r)[16]) const {
        Vec x[8], a[8], b[8];
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) x[i] = load(f + 8 * (p + i)), a[i] = load(q + 16 * (p + i)), b[i] = load(q + 16 * (p + i) + 8);
        transpose(x), transpose(a), transpose(b);
        const Factors s = entries(tables->roots, p);  // r[p + lane]
        Vec pa[4], pb[4], spa[4], spb[4];
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            const Vec t = times(x[i + 4], s);
            pa[i] = canonical(add(x[i], t)), pb[i] = canonical(diff(x[i], t));
            spa[i] = reduce(times(pa[i], s), kP), spb[i] = negate(reduce(times(pb[i], s), kP));
        }
        Vec ra[8], rb[8];
        leaf(pa, spa, a, ra);
        leaf(pb, spb, b, rb);
        transpose(ra), transpose(rb);
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) r[2 * i] = ra[i], r[2 * i + 1] = rb[i];
    }

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        for (std::size_t j = 0; j < count; j += 4, out += 128) {
            Vec r[16];
            products(2 * (first + j), r);
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i) {
                Vec g[4] = {r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]};
                inverse_h1(g, Group(tables->inverse_roots, first + j + i));
#pragma GCC unroll 4
                for (int t = 0; t < 4; ++t) store(out + 32 * i + 8 * t, g[t]);
            }
        }
    }
};

// Transforms of length a.size() with a bottom, as Transform's: forward in place of a[0, size),
// the rest zero and not read; inverse with output half, times scale.
template <class Bottom>
void forward_with(std::span<std::uint32_t> a, std::size_t size, const Tables& tables, const Bottom& bottom) {
    const Recursion recursion(tables.roots, tables.inverse_roots, bottom);
    const Source in(a.data(), size, 0);
    const std::size_t nv = a.size() / 8;
    auto* v = reinterpret_cast<Vec*>(a.data());
    if (std::countr_zero(nv) % 2 == 0) {
        const std::size_t h = nv / 4;
        forward_top4(in, v, h, tables.roots);
        for (std::size_t t = 0; t < 4; ++t) recursion.visit(a.data() + 8 * t * h, h, t);
    } else {
        const std::size_t h = nv / 2;
        forward_top2(in, v, h);
        recursion.visit(a.data(), h, 0);
        recursion.visit(a.data() + 8 * h, h, 1);
    }
}

template <class Bottom>
void inverse_with(std::span<std::uint32_t> a, const Tables& tables, const Bottom& bottom, std::uint32_t scale, Half output) {
    const Recursion recursion(tables.roots, tables.inverse_roots, bottom);
    const std::size_t nv = a.size() / 8;
    auto* v = reinterpret_cast<Vec*>(a.data());
    if (std::countr_zero(nv) % 2 == 0) {
        const std::size_t h = nv / 4;
        for (std::size_t t = 0; t < 4; ++t) recursion.visit(a.data() + 8 * t * h, h, t);
        inverse_top4(v, h, output, tables.inverse_roots, Factor(scale));
    } else {
        const std::size_t h = nv / 2;
        recursion.visit(a.data(), h, 0);
        recursion.visit(a.data() + 8 * h, h, 1);
        inverse_top2(v, h, output, scale);
    }
}

// Pruned transforms for the Kronecker layouts, in vectors of 8 coefficients. In a forward
// transform, vector bit b of the index (x = L, the padding bit) is zero in the input; the levels
// above b (y) act on each column separately, so they skip the columns with bit b set, and the
// level with bit b has half its inputs. In an inverse transform whose output is used only on the
// columns with bit b clear, the y levels compute those, the level with bit b half its outputs.
// Levels below b are full (Recursion), and so are small groups (kForwardFull, kInverseFull): the
// bottoms take at least 16 vectors, and narrow columns gain little.

// Butterflies j < h of radix-4 group k (inputs at j + t h) on the columns with bit b of j clear:
// kernels.hpp's loops (lib/ntt's butterflies), h >= 4 and h > 2^b.
inline void forward_columns(std::uint32_t* a, std::size_t h, std::size_t b, const std::uint32_t* roots, std::size_t k) {
    Vec* const v = reinterpret_cast<Vec*>(a);
    if (b == 0) return kernels::forward_even_columns(v, h, roots + slot(k), roots + slot(2 * k));
    kernels::forward_columns(v, h, std::size_t(1) << b, roots + slot(k), roots + slot(2 * k));
}

// The same with inverse butterflies.
inline void inverse_columns(std::uint32_t* a, std::size_t h, std::size_t b, const std::uint32_t* inverse_roots,
                            std::size_t k) {
    Vec* const v = reinterpret_cast<Vec*>(a);
    if (b == 0) return kernels::inverse_even_columns(v, h, inverse_roots + slot(k), inverse_roots + slot(2 * k));
    kernels::inverse_columns(v, h, std::size_t(1) << b, inverse_roots + slot(k), inverse_roots + slot(2 * k));
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
        if (low > b_) {
            forward_columns(a, h, b_, tables_.roots, k);
            for (std::size_t t = 0; t < 4; ++t) forward(a + 8 * t * h, h, 4 * k + t);
            return;
        }
        forward_half(a, h, low + 1 == b_, Group(tables_.roots, k));
        for (std::size_t t = 0; t < 4; ++t) recursion_.visit(a + 8 * t * h, h, 4 * k + t);
    }

    void inverse(std::uint32_t* a, std::size_t nv, std::size_t k) const {
        const std::size_t h = nv / 4, low = std::size_t(std::countr_zero(h));
        if (nv <= kInverseFull || low + 1 < b_) return recursion_.visit(a, nv, k);
        if (low > b_) {
            for (std::size_t t = 0; t < 4; ++t) inverse(a + 8 * t * h, h, 4 * k + t);
            return inverse_columns(a, h, b_, tables_.inverse_roots, k);
        }
        for (std::size_t t = 0; t < 4; ++t) recursion_.visit(a + 8 * t * h, h, 4 * k + t);
        inverse_half(a, h, low + 1 == b_, Group(tables_.inverse_roots, k));
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
// vector bit b clear (the others are left unspecified) of the output half; b lies below the top
// level's bits.
template <class Bottom>
void inverse_pruned(std::span<std::uint32_t> a, std::size_t b, const Tables& tables, const Bottom& bottom,
                    std::uint32_t scale, Half output = Half::kBoth) {
    const Pruned<Bottom> pruned(tables, bottom, b);
    const std::size_t nv = a.size() / 8, width = std::size_t(1) << b;
    auto* v = reinterpret_cast<Vec*>(a.data());
    const Factor s(scale);
    const auto scaled = [&s](Vec x) { return reduce(times(x, s), kP); };
    const bool lower = output != Half::kUpper, upper = output != Half::kLower;
    if (std::countr_zero(nv) % 2 == 0) {  // radix-4 identity group, as inverse_top4
        const std::size_t h = nv / 4;
        for (std::size_t t = 0; t < 4; ++t) pruned.inverse(a.data() + 8 * t * h, h, t);
        const Factor z(tables.inverse_roots[1], tables.inverse_roots[9]);
        for (std::size_t c = 0; c < h; c += 2 * width)
            for (std::size_t j = c; j < c + width; ++j) {
                const Vec x0 = v[j], x1 = v[j + h], x2 = v[j + 2 * h], x3 = v[j + 3 * h];
                const Vec ab = low(add(x0, x1)), cd = low(add(x2, x3));
                const Vec amb = low(diff(x0, x1)), cmd = times(diff(x2, x3), z);
                if (lower) v[j] = scaled(add(ab, cd)), v[j + h] = scaled(add(amb, cmd));
                if (upper) v[j + 2 * h] = scaled(diff(ab, cd)), v[j + 3 * h] = scaled(diff(amb, cmd));
            }
    } else {  // radix 2, as inverse_top2
        const std::size_t h = nv / 2;
        pruned.inverse(a.data(), h, 0);
        pruned.inverse(a.data() + 8 * h, h, 1);
        for (std::size_t c = 0; c < h; c += 2 * width)
            for (std::size_t j = c; j < c + width; ++j) {
                const Vec x0 = v[j], x1 = v[j + h];
                if (lower) v[j] = scaled(add(x0, x1));
                if (upper) v[j + h] = scaled(diff(x0, x1));
            }
    }
}

// Bump allocation of 64-byte aligned spans from scratch, with Arena's gaps: each take uses at
// most words(n) words.
class Carve {
public:
    explicit Carve(std::span<std::uint32_t> scratch) : rest_(scratch) {}

    static constexpr std::size_t words(std::size_t n) { return Arena::footprint(n) + 8; }

    std::span<std::uint32_t> take(std::size_t n) {
        const std::size_t skip = (0 - reinterpret_cast<std::uintptr_t>(rest_.data()) / 4) % 16;  // to 64 bytes
        if (skip + Arena::footprint(n) > rest_.size()) std::abort();
        const std::span<std::uint32_t> s = rest_.subspan(skip, n);
        rest_ = rest_.subspan(skip + Arena::footprint(n));
        return s;
    }

private:
    std::span<std::uint32_t> rest_;
};

// The levels of g and work spans. Level 0 holds the transform of length 2m of g, levels
// 1 .. T - 3 that of length 4m of Q_s, level T - 2 those of length m/2 of -q1, q2, -q3 for
// Q_(T-2) = 1 + x q1(y) + x^2 q2(y) + x^3 q3(y), level T - 1 that of length m of -q,
// Q_(T-1) = 1 + x q(y).
struct Levels {
    std::size_t m;  // 2^T >= 128
    int lg;         // T
    Tables tables;
    std::span<std::uint32_t> transforms;
    std::span<std::uint32_t> work[2];  // length 4m each

    std::span<std::uint32_t> level(int s) const { return transforms.subspan(std::size_t(s) * Arena::footprint(4 * m), 4 * m); }

    static std::size_t scratch(std::size_t m) {
        return Arena::footprint(Tables::words(m)) + Carve::words(std::size_t(std::countr_zero(m)) * Arena::footprint(4 * m)) +
               2 * Carve::words(4 * m);
    }

    Levels(std::size_t size, std::span<std::uint32_t> scratch)
        : m(size), lg(std::countr_zero(size)), tables(scratch.first(Tables::words(size)), size) {
        Carve carve(scratch.subspan(Arena::footprint(Tables::words(size))));
        transforms = carve.take(std::size_t(lg) * Arena::footprint(4 * m));
        for (auto& w : work) w = carve.take(4 * m);
    }
};

// From v = Q_s(x) Q_s(-x) mod (z^(2m) - 1) at stride L (2Y rows, wrapped: row 0 holds 1 + row 2Y) in
// a[0, 2m), Q_(s+1) at stride L in a[0, (2Y + 1) L): x below L / 2, rows 0 .. 2Y.
inline void next_level(std::span<std::uint32_t> a, std::size_t stride, std::size_t rows) {
    const std::size_t half = stride / 2, top = rows * stride;
    std::copy(a.begin(), a.begin() + std::ptrdiff_t(half), a.begin() + std::ptrdiff_t(top));
    a[top] = a[top] ? a[top] - 1 : kP - 1;
    std::fill(a.begin() + std::ptrdiff_t(top + half), a.begin() + std::ptrdiff_t(top + stride), 0);
    a[0] = 1;
    std::fill(a.begin() + 1, a.begin() + std::ptrdiff_t(half), 0);
    for (std::size_t i = 0; i < rows; ++i)
        std::fill(a.begin() + std::ptrdiff_t(i * stride + half), a.begin() + std::ptrdiff_t((i + 1) * stride), 0);
}

// The transforms of all levels of g. t: lg_max >= levels.lg + 1.
//  - Level 0, Q_0 = 1 - y g: G = T_2m(g); with u = x^2, g(x) g(-x) = v(u) from G by LevelBottom's
//    pairs, and Q_1 = 1 - 2y ge(u) + y^2 v(u) mod u^(m/2), g = ge(x^2) + x go(x^2).
//  - Levels 1 .. T - 3: Q_s at stride 2L has x below L (vector bit T - s - 3 zero: pruned forward);
//    Q_(s+1) keeps v at stride L for x below L / 2 (bit T - s - 4: pruned inverse).
//  - Level T - 3 (stride 8): Q_(T-2) = 1 + x q1 + x^2 q2 + x^3 q3, deg q_k <= m/4, from v at stride 8:
//    q_k[i] = v[8i + k] for 0 < i < m/4 and the wrapped q_k[m/4] = v[k]. Then Q_(T-1) = 1 + x q with
//    q = 2 q2 - q1^2 (deg <= m/2; q1^2 has no y^0 term, so its y^(m/2) wraps onto y^0).
inline void build_levels(const Transform& t, std::span<const std::uint32_t> g, Levels& levels) {
    const std::size_t m = levels.m;
    const std::span<std::uint32_t> v = levels.work[0].first(2 * m), q0 = levels.level(0).first(2 * m);
    std::fill(q0.begin(), q0.end(), 0);
    std::copy(g.begin() + std::ptrdiff_t(std::min<std::size_t>(g.size(), 1)), g.begin() + std::ptrdiff_t(std::min(g.size(), m)),
              q0.begin() + 1);
    forward_with(q0, m, levels.tables, LevelBottom{&levels.tables, v.data()});
    t.inverse(v.first(m));
    const std::span<std::uint32_t> q1 = levels.level(1).first(3 * m);  // stride m: rows 1, -2 ge, v
    std::fill(q1.begin(), q1.end(), 0);
    q1[0] = 1;
    for (std::size_t j = 0; j < m / 2; ++j) q1[m + j] = minus_twice(j > 0 && 2 * j < g.size() ? g[2 * j] : 0), q1[2 * m + j] = v[j];
    const Tables& tables = levels.tables;
    for (int s = 1; s + 2 < levels.lg; ++s) {  // Q_s: rows 0 .. Y at stride 2L
        const std::size_t stride = 2 * (m >> s), pad = std::size_t(levels.lg - s - 3);
        forward_pruned(levels.level(s), ((std::size_t(1) << s) + 1) * stride, pad, tables, LevelBottom{&tables, v.data()});
        if (s + 3 < levels.lg) {
            const std::span<std::uint32_t> next = levels.level(s + 1).first(2 * m);
            inverse_pruned(next, pad - 1, tables, InverseBottom{tables.inverse_roots, v.data()}, kInverseScales[0][levels.lg + 1]);
            next_level(levels.level(s + 1), stride / 2, std::size_t(2) << s);
        }
    }
    t.inverse(v);  // stride 8
    const std::span<std::uint32_t> q = levels.level(levels.lg - 1).first(m);  // -q_(T-1)
    std::span<std::uint32_t> c[4];
    for (std::size_t k = 1; k < 4; ++k) {  // c[k] = (-1)^k q_k
        c[k] = levels.level(levels.lg - 2).subspan((k - 1) * Arena::footprint(m / 2), m / 2);
        std::fill(c[k].begin(), c[k].end(), 0);
        for (std::size_t i = 1; i <= m / 4; ++i) {
            const std::uint32_t x = v[8 * (i % (m / 4)) + k];
            c[k][i] = k % 2 && x ? kP - x : x;
        }
    }
    std::fill(q.begin(), q.end(), 0);
    for (std::size_t i = 1; i <= m / 4; ++i) q[i] = minus_twice(c[2][i]);
    const std::span<std::uint32_t> square = levels.work[1].first(m / 2);  // q1^2 mod (y^(m/2) - 1)
    for (std::size_t k = 1; k < 4; ++k) t.forward(c[k]);
    t.inverse_product(c[1], c[1], square);
    for (std::size_t i = 1; i <= m / 2; ++i) {
        const std::uint32_t x = q[i] + square[i % (m / 2)];
        q[i] = x >= kP ? x - kP : x;
    }
    t.forward(q);
}

}  // namespace detail

// Transform length compose() uses for n coefficients: the Transform needs lg_max >= this.
inline int compose_log(std::size_t n) {
    if (n <= detail::kComposeBase) return Transform::kMinLog;
    return int(std::bit_width(std::max<std::size_t>(n, 128) - 1)) + 1;
}

// Scratch words for compose() of n coefficients.
inline std::size_t compose_scratch(std::size_t n) {
    if (n <= detail::kComposeBase) return 0;
    return detail::Levels::scratch(std::bit_ceil(std::max<std::size_t>(n, 128)));
}

// h = f(g) mod x^n for n = h.size() >= 1 and g[0] = 0 (if g is not empty); coefficients past
// f.size() and g.size() are zero. scratch: compose_scratch(n) words, 32-byte aligned (from an
// Arena). t: lg_max >= compose_log(n). f, g and h must not overlap.
//
// Transposed power projection: with m = 2^T >= n, the transpose of w -> ([x^(m-1)] w g^i)_i maps f
// to h reversed. Level s maps P_(s+1) (L / 2 by 2Y coefficients) to P_s (L by Y):
// P_s[a][b] = sum P_(s+1)[(a + c - 1) / 2][b + e] Q_s(-x)[c][e] over odd a + c. With P_(s+1)
// reversed in x and y (stored at stride L), this is R = P(z^2) Q_s(-z) mod (z^4m - 1): P_s[a][b] =
// R[L - 1 - a][2Y - 1 - b], so rows Y .. 2Y - 1 of R, x below L, are P_s reversed in x and y, the
// input of level s - 1. P_T = f (one row in x); h[k] = P_0[m - 1 - k][0] = R[k][1] at level 0.
// The first and the last two levels are one-dimensional:
//  - level T - 1: P = f reversed, R = p(y) (1 - x q(y)): rows y in [m/2, m) of p and -p q;
//  - level T - 2: P = p0(y) + x^2 p1(y), R = P Q_(T-2)(-x) mod x^4 by columns;
//  - level 0: P = p0(x) + y p1(x), R's row 1 = p1(x^2) - p0(x^2) g(-x) = h.
// Level s's transforms are pruned (composition.hpp's levels): P_(s+1) at stride L has x below L / 2
// (vector bit T - s - 4 zero, but level T - 3: 4 of 8 words), and of R only x below L is used.
inline void compose(const Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                    std::span<std::uint32_t> h, std::span<std::uint32_t> scratch) {
    const std::size_t n = h.size();
    if (n <= detail::kComposeBase) return detail::compose_direct(f, g, h);
    detail::Levels levels(std::bit_ceil(std::max<std::size_t>(n, 128)), scratch);
    detail::build_levels(t, g, levels);
    const std::size_t m = levels.m;
    std::span<std::uint32_t> in = levels.work[0], out = levels.work[1];
    // Level T - 1: columns p0 = p[m/2, m), p1 = (-p q)[m/2, m) of P_(T-2).
    const std::span<std::uint32_t> p = out.first(m), pq = out.subspan(m, m);
    for (std::size_t i = 0; i < m; ++i) p[i] = m - 1 - i < f.size() ? f[m - 1 - i] : 0;
    t.cyclic_product(p, 0, pq, levels.level(levels.lg - 1).first(m), Half::kUpper);
    // Level T - 2, length m/2: columns r0 = p0, r1 = -p0 q1, r2 = p1 + p0 q2, r3 = -p1 q1 - p0 q3,
    // rows [m/4, m/2), into in[2m, 4m) at stride 8 (columns 4 .. 7 zero).
    {
        const std::size_t h = m / 2;
        const std::span<std::uint32_t> p0 = p.subspan(h), p1 = pq.subspan(h), t0 = in.first(h), t1 = in.subspan(m, h);
        const std::span<std::uint32_t> c = levels.level(levels.lg - 2);
        const auto column = [&](std::size_t k) { return c.subspan((k - 1) * Arena::footprint(h), h); };
        const std::span<std::uint32_t> r1 = out.subspan(2 * m, h), r3 = out.subspan(2 * m + h, h), r2 = out.subspan(3 * m, h);
        t.forward(p0, 0, t0);
        t.forward(p1, 0, t1);
        t.inverse_product(t0, column(1), r1, Half::kUpper);
        t.inverse_product(t0, column(2), r2, Half::kUpper);
        const Transform::Pair pairs[2] = {{t1, column(1)}, {t0, column(3)}};
        t.inverse_product_sum(pairs, r3, Half::kUpper);
        for (std::size_t i = h / 2; i < h; ++i) {
            std::uint32_t* row = &in[2 * m + 8 * (i - h / 2)];
            const std::uint32_t x = p1[i] + r2[i], y = r3[i];
            row[0] = p0[i], row[1] = r1[i], row[2] = x >= kModulus ? x - kModulus : x, row[3] = y;
            std::fill(row + 4, row + 8, 0);
        }
    }
    const std::uint32_t scale = detail::kInverseScales[1][levels.lg + 2];
    for (int s = levels.lg - 3; s >= 1; --s) {
        const std::span<std::uint32_t> x = in.subspan(2 * m);
        const std::size_t pad = std::size_t(levels.lg - s - 3);
        if (s == levels.lg - 3) t.forward(x);
        else detail::forward_pruned(x, 2 * m, pad - 1, levels.tables, detail::ForwardBottom{levels.tables.roots});
        detail::inverse_pruned(out, pad, levels.tables, detail::CompositionBottom{&levels.tables, x.data(), levels.level(s).data()},
                               scale, Half::kUpper);
        std::swap(in, out);
    }
    // Level 0: p0 = in[2m, 3m), p1 = in[3m, 4m) (x below m/2).
    const std::span<std::uint32_t> p0 = in.subspan(2 * m, m), r = out.first(2 * m);
    t.forward(p0.first(m / 2), 0, p0);
    detail::inverse_with(r, levels.tables, detail::CompositionBottom{&levels.tables, p0.data(), levels.level(0).data()},
                         detail::kInverseScales[1][levels.lg + 1], Half::kLower);
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t a = k % 2 ? 0 : in[3 * m + k / 2];
        h[k] = a >= r[k] ? a - r[k] : a + kModulus - r[k];
    }
}

}  // namespace poly
