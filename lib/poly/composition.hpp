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
// and degree Y = 2^s in y; Q_s(x, 0) = Q_s(0, y) = 1. Level s stores the transform of length 4m of
// Q_s by Kronecker substitution x = z, y = z^(2L): x below L, rows 0 .. Y of 2Y. A product mod
// (z^4m - 1) then carries nothing from x into y while the x degree stays below 2L, and wraps y
// mod y^(2Y).
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

// -x mod P in [0, P) for x < P.
inline Vec negate(Vec x) {
    return _mm256_min_epu32(_mm256_sub_epi32(broadcast(kP), x), _mm256_sub_epi32(_mm256_setzero_si256(), x));
}

// x w mod P in [0, 2P) for any x < 2^32 and a factor per lane (times() needs one factor).
inline Vec times_lanes(Vec x, const Factor& f) {
    const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(x, f.q), 32);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(f.q, 32));
    const Vec q = _mm256_blend_epi32(even, odd, 0xAA);
    return _mm256_sub_epi32(_mm256_mullo_epi32(x, f.w), _mm256_mullo_epi32(q, broadcast(kP)));
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
// times 2^-32, in [0, 2P). Every sum stays below 8 P^2.
inline void leaf_graeffe(const Vec (&c)[8], Vec s_m, Vec (&out)[4]) {
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
    static void leaf(const Vec (&p)[4], const Vec (&wp)[4], const Vec (&c)[8], Vec (&out)[8]) {
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
        Factor s(0);  // r[p + lane]
        s.w = load(tables->roots + slot(p)), s.q = load(tables->roots + slot(p) + 8);
        Vec pa[4], pb[4], spa[4], spb[4];
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            const Vec t = times_lanes(x[i + 4], s);
            pa[i] = canonical(add(x[i], t)), pb[i] = canonical(diff(x[i], t));
            spa[i] = reduce(times_lanes(pa[i], s), kP), spb[i] = negate(reduce(times_lanes(pb[i], s), kP));
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

// Transforms of length a.size() with a bottom, as Transform's: forward in place (input a);
// inverse with output half, times scale.
template <class Bottom>
void forward_with(std::span<std::uint32_t> a, const Tables& tables, const Bottom& bottom) {
    const Recursion recursion(tables.roots, tables.inverse_roots, bottom);
    const Source in(a.data(), a.size(), 0);
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
// a[0, 2m), Q_(s+1) at stride L in a: x below L / 2, rows 0 .. 2Y of 4Y.
inline void next_level(std::span<std::uint32_t> a, std::size_t stride, std::size_t rows) {
    const std::size_t half = stride / 2, top = rows * stride;
    std::copy(a.begin(), a.begin() + std::ptrdiff_t(half), a.begin() + std::ptrdiff_t(top));
    a[top] = a[top] ? a[top] - 1 : kP - 1;
    std::fill(a.begin() + std::ptrdiff_t(top + half), a.end(), 0);
    a[0] = 1;
    std::fill(a.begin() + 1, a.begin() + std::ptrdiff_t(half), 0);
    for (std::size_t i = 0; i < rows; ++i)
        std::fill(a.begin() + std::ptrdiff_t(i * stride + half), a.begin() + std::ptrdiff_t((i + 1) * stride), 0);
}

// The transforms of all levels of g. t: lg_max >= levels.lg + 1.
//  - Level 0, Q_0 = 1 - y g: G = T_2m(g); with u = x^2, g(x) g(-x) = v(u) from G by LevelBottom's
//    pairs, and Q_1 = 1 - 2y ge(u) + y^2 v(u) mod u^(m/2), g = ge(x^2) + x go(x^2).
//  - Level T - 3 (stride 8): Q_(T-2) = 1 + x q1 + x^2 q2 + x^3 q3, deg q_k <= m/4, from v at stride 8:
//    q_k[i] = v[8i + k] for 0 < i < m/4 and the wrapped q_k[m/4] = v[k]. Then Q_(T-1) = 1 + x q with
//    q = 2 q2 - q1^2 (deg <= m/2; q1^2 has no y^0 term, so its y^(m/2) wraps onto y^0).
inline void build_levels(const Transform& t, std::span<const std::uint32_t> g, Levels& levels) {
    const std::size_t m = levels.m;
    const std::span<std::uint32_t> v = levels.work[0].first(2 * m), q0 = levels.level(0).first(2 * m);
    std::fill(q0.begin(), q0.end(), 0);
    std::copy(g.begin() + std::ptrdiff_t(std::min<std::size_t>(g.size(), 1)), g.begin() + std::ptrdiff_t(std::min(g.size(), m)),
              q0.begin() + 1);
    forward_with(q0, levels.tables, LevelBottom{&levels.tables, v.data()});
    t.inverse(v.first(m));
    const std::span<std::uint32_t> q1 = levels.level(1);
    std::fill(q1.begin(), q1.end(), 0);  // stride m: rows 1, -2 ge, v
    q1[0] = 1;
    for (std::size_t j = 0; j < m / 2; ++j) {
        const std::uint32_t e = j > 0 && 2 * j < g.size() ? g[2 * j] : 0, minus_2e = e ? 2 * kP - 2 * e : 0;  // < 2P
        q1[m + j] = minus_2e >= kP ? minus_2e - kP : minus_2e;
        q1[2 * m + j] = v[j];
    }
    for (int s = 1; s + 2 < levels.lg; ++s) {
        forward_with(levels.level(s), levels.tables, LevelBottom{&levels.tables, v.data()});
        if (s + 3 < levels.lg) {
            t.inverse(v, levels.level(s + 1).first(2 * m));
            next_level(levels.level(s + 1), m >> s, std::size_t(2) << s);
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
    for (std::size_t i = 1; i <= m / 4; ++i) q[i] = c[2][i] ? 2 * kP - 2 * c[2][i] - (2 * c[2][i] <= kP ? kP : 0) : 0;
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
inline void compose(const Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                    std::span<std::uint32_t> h, std::span<std::uint32_t> scratch) {
    using ntt::detail::multiply_mod, ntt::detail::power;
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
    const std::uint32_t scale = multiply_mod(power(std::uint32_t(m / 2), kModulus - 2), ntt::detail::kR);
    for (int s = levels.lg - 3; s >= 1; --s) {
        const std::span<std::uint32_t> x = in.subspan(2 * m);
        t.forward(x);
        detail::inverse_with(out, levels.tables, detail::CompositionBottom{&levels.tables, x.data(), levels.level(s).data()},
                             scale, Half::kUpper);
        const std::size_t stride = 2 * (m >> s);  // rows of R; level s - 1 reads x below stride / 2
        for (std::size_t i = 2 * m; i < 4 * m; i += stride)
            std::fill(out.begin() + std::ptrdiff_t(i + stride / 2), out.begin() + std::ptrdiff_t(i + stride), 0);
        std::swap(in, out);
    }
    // Level 0: p0 = in[2m, 3m), p1 = in[3m, 4m) (x below m/2).
    const std::span<std::uint32_t> p0 = in.subspan(2 * m, m), r = out.first(2 * m);
    t.forward(p0);
    detail::inverse_with(r, levels.tables, detail::CompositionBottom{&levels.tables, p0.data(), levels.level(0).data()},
                         multiply_mod(power(std::uint32_t(m / 4), kModulus - 2), ntt::detail::kR), Half::kLower);
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t a = k % 2 ? 0 : in[3 * m + k / 2];
        h[k] = a >= r[k] ? a - r[k] : a + kModulus - r[k];
    }
}

}  // namespace poly
