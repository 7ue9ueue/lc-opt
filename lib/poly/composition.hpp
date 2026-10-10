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
// and degree Y = 2^s in y; Q_s(x, 0) = Q_s(0, y) = 1. A level 1 < s < T - 2 is stored as the
// transform of length 4m of Q_s by Kronecker substitution x = z, y = z^(2L): x below L, rows
// 0 .. Y of 2Y. A product mod (z^4m - 1) then carries nothing from x into y while the x degree
// stays below 2L, and wraps y mod y^(2Y). Levels 0, 1, T - 2 and T - 1 are one-dimensional (Levels).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// Up to this many coefficients, Horner's rule.
inline constexpr std::size_t kComposeBase = 32;

// m = 2^T for n coefficients: at least 256, so that level 1's inverse transforms (length m) have
// tiles of 16 vectors, as CompositionSumBottom needs.
inline std::size_t compose_length(std::size_t n) { return std::bit_ceil(std::max<std::size_t>(n, 256)); }

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

// Leaf sums in transposed form (one leaf per lane), over one parity of lanes: each 64-bit lane
// multiplies the low 32 bits (the even lanes as given, the odd lanes after odd_lanes).
inline Vec product(Vec x, Vec y) { return _mm256_mul_epu32(x, y); }
inline Vec twice(Vec x) { return _mm256_add_epi64(x, x); }
inline Vec plus(Vec x, Vec y) { return _mm256_add_epi64(x, y); }
inline Vec minus(Vec x, Vec y) { return _mm256_sub_epi64(x, y); }

// 4 P^2 in each 64-bit lane: the offset of a sum with up to 4 negative terms.
inline Vec offset() { return _mm256_set1_epi64x(static_cast<long long>(4 * std::uint64_t(kP) * kP)); }

// The odd lanes of each vector moved to the even ones.
template <std::size_t N>
[[gnu::always_inline]] inline void odd_lanes(const Vec (&x)[N], Vec (&out)[N]) {
#pragma GCC unroll 8
    for (std::size_t i = 0; i < N; ++i) out[i] = _mm256_srli_epi64(x[i], 32);
}

// x / 2^32 mod P in [0, x / 2^32 + P) per lane, from the 64-bit sums x of the even lanes and of the
// odd lanes, each below 14 P^2 (Montgomery).
inline Vec reduce_sums(Vec even, Vec odd) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
    even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
    odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
    return _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
}

// c = Q mod (z^8 - t) = e(u) + z o(u), u = z^2 (coefficient k in c[k], canonical), and
// tc[k] = t c[k] for k >= 4 (canonical): sum = e^2 - u o^2 mod (u^4 - t), the leaf of Q(z) Q(-z),
// as 64-bit sums. Each has 4 negative terms, offset by 4 P^2, and stays below 8 P^2.
[[gnu::always_inline]] inline void graeffe_sums(const Vec (&c)[8], const Vec (&tc)[8], Vec (&sum)[4]) {
    const auto x = [&c](int i, int j) { return product(c[i], c[j]); };
    const auto t = [&c, &tc](int i, int j) { return product(c[i], tc[j]); };
    sum[0] = minus(minus(plus(plus(x(0, 0), twice(t(2, 6))), plus(t(4, 4), offset())), twice(t(1, 7))), twice(t(3, 5)));
    sum[1] = minus(minus(plus(plus(twice(x(0, 2)), twice(t(4, 6))), offset()), x(1, 1)), plus(twice(t(3, 7)), t(5, 5)));
    sum[2] = minus(minus(plus(plus(twice(x(0, 4)), x(2, 2)), plus(t(6, 6), offset())), twice(x(1, 3))), twice(t(5, 7)));
    sum[3] = minus(minus(plus(plus(twice(x(0, 6)), twice(x(2, 4))), offset()), twice(x(1, 5))), plus(x(3, 3), t(7, 7)));
}

// The sums over the even lanes, then over the odd lanes, reduced: times 2^-32, in [0, 2P).
[[gnu::always_inline]] inline void graeffe_leaf(const Vec (&c)[8], const Vec (&tc)[8], Vec (&out)[4]) {
    Vec even[4], odd[4], c2[8], tc2[8];
    graeffe_sums(c, tc, even);
    odd_lanes(c, c2), odd_lanes(tc, tc2);
    graeffe_sums(c2, tc2, odd);
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) out[k] = low(reduce_sums(even[k], odd[k]));
}

// Leaves 2 (k + i) (or 2 (k + i) + 1 at from + 8) for i < 8 of a transform, transposed: c[j] holds
// coefficient j of the 8 leaves.
[[gnu::always_inline]] inline void load_leaves(const std::uint32_t* from, Vec (&c)[8]) {
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) c[i] = load(from + 16 * i);
    transpose(c);
}

// tc[j] = t c[j] for j >= first (canonical), t = s per lane, or -s if kNegative; the rest zero.
template <bool kNegative>
[[gnu::always_inline]] inline void scaled_leaves(const Vec (&c)[8], const Factors& s, int first, Vec (&tc)[8]) {
#pragma GCC unroll 8
    for (int j = 0; j < 8; ++j) {
        const Vec x = reduce(times(c[j], s), kP);
        tc[j] = j < first ? _mm256_setzero_si256() : kNegative ? negate(x) : x;
    }
}

// Leaves k .. k + 7 at out from x mod (u^4 - s) and y mod (u^4 + s): (x + y) + u^4 (x - y) / s,
// twice their CRT, in [0, 2P).
inline void combine_pairs(const Vec (&x)[4], const Vec (&y)[4], const Factors& inverse, std::uint32_t* out) {
    Vec r[8];
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) r[i] = low(add(x[i], y[i])), r[i + 4] = times(diff(x[i], y[i]), inverse);
    transpose(r);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) store(out + 8 * i, r[i]);
}

// Twiddle tables for transforms of length 4m. Pair p of their leaves (leaves 2p, 2p + 1) has the
// moduli z^8 -+ s for s = r[p], entry p of roots.
struct Tables {
    std::uint32_t *roots, *inverse_roots;

    static std::size_t words(std::size_t m) { return 2 * Arena::footprint(ntt::detail::table_words(std::countr_zero(m) + 2)); }

    Tables(std::span<std::uint32_t> memory, std::size_t m) {
        roots = memory.data(), inverse_roots = roots + Arena::footprint(ntt::detail::table_words(std::countr_zero(m) + 2));
        ntt::detail::build_table(roots, m / 4, ntt::detail::kRoots[0]);
        ntt::detail::build_table(inverse_roots, m / 4, ntt::detail::kRoots[1]);
    }
};

// s = r[k + lane] and its inverse, for 8 pairs from pair k (a multiple of 8).
struct PairWeights {
    Factors s, inverse;
    PairWeights(const Tables& tables, std::size_t k) : s(entries(tables.roots, k)), inverse(entries(tables.inverse_roots, k)) {}
};

// a = P mod (z^8 - t), c and tc as for graeffe_sums: sum = the even (parity 0) or odd (parity 1)
// part of a(z) c(-z) mod (z^8 - t), as a series in u mod (u^4 - t) (the odd part divided by z):
// sum_k adds (-1)^j a_i c_j over i + j = 2k + parity and (-1)^j a_i tc_j over i + j = 2k + 8 +
// parity (tc[j] for j >= 1 - parity). 8 terms, 4 of them negative (offset by 4 P^2), below 8 P^2.
template <int kParity>
[[gnu::always_inline]] inline void product_sums(const Vec (&a)[8], const Vec (&c)[8], const Vec (&tc)[8],
                                                Vec (&sum)[4]) {
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
        Vec s = offset();
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) {
            const int j = 2 * k + kParity - i, l = j + 8;
            if (j >= 0) s = j % 2 ? minus(s, product(a[i], c[j])) : plus(s, product(a[i], c[j]));
            if (l < 8) s = l % 2 ? minus(s, product(a[i], tc[l])) : plus(s, product(a[i], tc[l]));
        }
        sum[k] = s;
    }
}

// The sums over the even lanes, then over the odd lanes, reduced: times 2^-32, in [0, 2P).
template <int kParity>
[[gnu::always_inline]] inline void product_leaf(const Vec (&a)[8], const Vec (&c)[8], const Vec (&tc)[8],
                                                Vec (&out)[4]) {
    Vec even[4], odd[4], a2[8], c2[8], tc2[8];
    product_sums<kParity>(a, c, tc, even);
    odd_lanes(a, a2), odd_lanes(c, c2), odd_lanes(tc, tc2);
    product_sums<kParity>(a2, c2, tc2, odd);
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) out[k] = low(reduce_sums(even[k], odd[k]));
}

// Level 1's products (composition and power projection) for 8 leaf pairs from pair k of the
// transforms of q1, q2 (and p1 if kNumerator) at length m: G(q1), G(q2), E(q1, q2) (and O(p1, q1),
// O(p1, q2)), where G(a) = a(x) a(-x) and E(a, b), O(a, b) are the even and the odd part of
// a(x) b(-x), all series in u = x^2. Leaves k .. k + 7 of transforms of length m/2 at out[i] + 8k,
// times 2^-31.
template <bool kNumerator>
struct FirstLevelProducts {
    static constexpr int kOutputs = kNumerator ? 5 : 3;
    const Tables* tables;
    const std::uint32_t *q1, *q2, *p1;
    std::uint32_t* out[kOutputs];

    [[gnu::flatten]] void pairs(std::size_t k) const {
        const PairWeights weights(*tables, k);
        Vec a[kOutputs][4], b[kOutputs][4];
        leaves<false>(16 * k, weights.s, a);
        leaves<true>(16 * k + 8, weights.s, b);
#pragma GCC unroll 5
        for (int i = 0; i < kOutputs; ++i) combine_pairs(a[i], b[i], weights.inverse, out[i] + 8 * k);
    }

    template <bool kNegative>
    [[gnu::noinline, gnu::flatten]] void leaves(std::size_t at, const Factors& s, Vec (&r)[kOutputs][4]) const {
        Vec c1[8], c2[8], a[8], t1[8], t2[8];
        load_leaves(q1 + at, c1), load_leaves(q2 + at, c2);
        if constexpr (kNumerator) load_leaves(p1 + at, a);
        scaled_leaves<kNegative>(c1, s, kNumerator ? 2 : 4, t1);
        scaled_leaves<kNegative>(c2, s, 1, t2);
        graeffe_leaf(c1, t1, r[0]);
        graeffe_leaf(c2, t2, r[1]);
        product_leaf<0>(c1, c2, t2, r[2]);
        if constexpr (kNumerator) {
            product_leaf<1>(a, c1, t1, r[3]);
            product_leaf<1>(a, c2, t2, r[4]);
        }
    }
};

// The scale of an inverse transform of length 2^lg of LevelBottom's output: (2^lg / 8)^-1 2^31.
inline std::uint32_t graeffe_scale(int lg) {
    return ntt::detail::multiply_mod(kInverseScales[0][lg], std::uint32_t((std::uint64_t(1) << 31) % kP));
}

// The bottom of the forward transform of Q_s (length 4m): forward butterflies (canonical leaves,
// kept as the level), then, unless graeffe is null, the transform of length 2m of
// V = Q_s(x) Q_s(-x) at stride L (2Y rows, wrapped), times 2^-31 (graeffe_scale undoes it). For
// leaves a = Q mod (z^8 - s), b = Q mod (z^8 + s): V mod (u^4 -+ s) = A, B = a(z) a(-z), b(z) b(-z)
// (graeffe_leaf, u = z^2), and leaf k of V's transform is V mod (u^8 - s^2), twice it
// (A + B) + u^4 (A - B) / s. Pairs go 8 at a time (two groups), one per lane, after their leaves
// are read, so graeffe may be the transform itself (each pair writes below the leaves read so far);
// tiles hold at least 16 vectors (4m >= 256).
struct LevelBottom {
    static constexpr bool kForward = true, kInverse = false;
    const Tables* tables;
    std::uint32_t* graeffe;

    void operator()(std::uint32_t* a, std::size_t count, std::size_t first) const {
        ForwardBottom{tables->roots}(a, count, first);
        if (graeffe)
            for (std::size_t i = 0; i < count; i += 4) pairs(a + 32 * i, 2 * (first + i));
    }

    // 8 pairs from leaves at q, the first pair k.
    [[gnu::flatten]] void pairs(const std::uint32_t* q, std::size_t k) const {
        const PairWeights weights(*tables, k);
        Vec va[4], vb[4];
        leaves<false>(q, weights.s, va);
        leaves<true>(q + 8, weights.s, vb);
        combine_pairs(va, vb, weights.inverse, graeffe + 8 * k);
    }

    // V mod (u^4 - t) for t = s (or -s if kNegative) of the 8 leaves at q, times 2^-32.
    template <bool kNegative>
    [[gnu::noinline, gnu::flatten]] static void leaves(const std::uint32_t* q, const Factors& s, Vec (&v)[4]) {
        Vec c[8], tc[8];
        load_leaves(q, c);
        scaled_leaves<kNegative>(c, s, 4, tc);
        graeffe_leaf(c, tc, v);
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

    // One parity of lanes, p[i] = p_i and wp[i] = w p_i (canonical), c as above: sum[k] =
    // coefficient 2k + kParity of R mod (z^8 - w) as a 64-bit sum of u^i c_(2j + kParity) over
    // i + j = k (times p_i) and i + j = k + 4 (times w p_i); for kParity = 1 negated, from 4 P^2.
    // Below 4 P^2.
    template <int kParity>
    [[gnu::always_inline]] static void sums(const Vec (&p)[4], const Vec (&wp)[4], const Vec (&c)[8], Vec (&sum)[4]) {
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            const Vec first = product(p[0], c[2 * k + kParity]);
            Vec s = kParity ? minus(offset(), first) : first;
#pragma GCC unroll 3
            for (int i = 1; i < 4; ++i) {
                const Vec term = product(i <= k ? p[i] : wp[i], c[2 * ((k - i + 4) % 4) + kParity]);
                s = kParity ? minus(s, term) : plus(s, term);
            }
            sum[k] = s;
        }
    }

    // One leaf per lane, p, wp and c as above: out[k] = coefficient k of R mod (z^8 - w), times
    // 2^-32, in [0, 2P).
    [[gnu::always_inline]] static void leaf(const Vec (&p)[4], const Vec (&wp)[4], const Vec (&c)[8], Vec (&out)[8]) {
        Vec p2[4], wp2[4], c2[8], even[4], odd[4];
        odd_lanes(p, p2), odd_lanes(wp, wp2), odd_lanes(c, c2);
        sums<0>(p, wp, c, even);
        sums<0>(p2, wp2, c2, odd);
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) out[2 * k] = reduce_sums(even[k], odd[k]);
        sums<1>(p, wp, c, even);
        sums<1>(p2, wp2, c2, odd);
#pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) out[2 * k + 1] = reduce_sums(even[k], odd[k]);
    }

    // Leaves 2p .. 2p + 15 of R into r.
    [[gnu::flatten]] void products(std::size_t p, Vec (&r)[16]) const {
        Vec x[8], c[8], pa[4], pb[4], spa[4], spb[4], ra[8], rb[8];
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) x[i] = load(f + 8 * (p + i));
        transpose(x);
        const Factors s = entries(tables->roots, p);  // r[p + lane]
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            const Vec t = times(x[i + 4], s);
            pa[i] = canonical(add(x[i], t)), pb[i] = canonical(diff(x[i], t));
            spa[i] = reduce(times(pa[i], s), kP), spb[i] = negate(reduce(times(pb[i], s), kP));
        }
        load_leaves(q + 16 * p, c);
        leaf(pa, spa, c, ra);
        load_leaves(q + 16 * p + 8, c);
        leaf(pb, spb, c, rb);
        transpose(ra), transpose(rb);
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) r[2 * i] = ra[i], r[2 * i + 1] = rb[i];
    }

    // Inverse butterflies of the leaves r of groups g .. g + 3 into out.
    [[gnu::always_inline]] static void finish(const Tables& tables, const Vec (&r)[16], std::size_t g, std::uint32_t* out) {
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            Vec f[4] = {r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]};
            inverse_h1(f, Group(tables.inverse_roots, g + i));
#pragma GCC unroll 4
            for (int t = 0; t < 4; ++t) store(out + 32 * i + 8 * t, f[t]);
        }
    }

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        for (std::size_t j = 0; j < count; j += 4, out += 128) {
            Vec r[16];
            products(2 * (first + j), r);
            finish(*tables, r, first + j, out);
        }
    }
};

// The same for R = P1(z^2) Q1(-z) + P2(z^2) Q2(-z): the leaves of both terms, added.
struct CompositionSumBottom {
    static constexpr bool kForward = false, kInverse = true;
    CompositionBottom terms[2];

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        for (std::size_t j = 0; j < count; j += 4, out += 128) {
            Vec r[16], r2[16];
            terms[0].products(2 * (first + j), r);
            terms[1].products(2 * (first + j), r2);
#pragma GCC unroll 16
            for (int i = 0; i < 16; ++i) r[i] = low(add(r[i], r2[i]));
            CompositionBottom::finish(*terms[0].tables, r, first + j, out);
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

// The levels of g and work spans. Level 0 holds the transform of length 2m of g, level 1 those
// of length m of q1 and q2 for Q_1 = 1 + y q1(x) + y^2 q2(x) (first_transform), levels
// 2 .. T - 4 that of length 4m of Q_s, level T - 3 those of length m/4 of (-1)^k q_k for
// Q_(T-3) = sum_(k < 8) x^k q_k(y) (column_transform), level T - 2 those of length m/2 of -q1, q2, -q3 for
// Q_(T-2) = 1 + x q1(y) + x^2 q2(y) + x^3 q3(y), level T - 1 that of length m of -q,
// Q_(T-1) = 1 + x q(y). Each level has a slot of 4m words; temporaries live in slots that are not
// filled yet (forward pass) or no longer read (backward pass).
struct Levels {
    std::size_t m;  // 2^T >= 256
    int lg;         // T
    Tables tables;
    std::span<std::uint32_t> transforms;

    std::span<std::uint32_t> level(int s) const { return transforms.subspan(std::size_t(s) * Arena::footprint(4 * m), 4 * m); }
    // Level 1's transform of q1 (i = 0) or q2 (i = 1).
    std::span<std::uint32_t> first_transform(std::size_t i) const { return level(1).subspan(i * Arena::footprint(m), m); }
    // Level T - 3's transform of c_k, 1 <= k < 8 (third_last_level).
    std::span<std::uint32_t> column_transform(std::size_t k) const {
        return level(lg - 3).subspan((k - 1) * Arena::footprint(m / 4), m / 4);
    }

    static std::size_t scratch(std::size_t m) {
        return Arena::footprint(Tables::words(m)) + Carve::words(std::size_t(std::countr_zero(m)) * Arena::footprint(4 * m));
    }

    Levels(std::size_t size, std::span<std::uint32_t> scratch)
        : m(size), lg(std::countr_zero(size)), tables(scratch.first(Tables::words(size)), size) {
        Carve carve(scratch.subspan(Arena::footprint(Tables::words(size))));
        transforms = carve.take(std::size_t(lg) * Arena::footprint(4 * m));
    }
};

// From v = Q_s(x) Q_s(-x) mod (z^(2m) - 1) at stride L (2Y rows, wrapped: row 0 holds 1 + row 2Y) in
// a[0, 2m), Q_(s+1) at stride L in a[0, (2Y + 1) L): rows 0 .. 2Y, x below L / 2. The rest of each
// row is left as it is: the pruned forward transform does not read it.
inline void next_level(std::span<std::uint32_t> a, std::size_t stride, std::size_t rows) {
    const std::size_t half = stride / 2, top = rows * stride;
    std::copy(a.begin(), a.begin() + std::ptrdiff_t(half), a.begin() + std::ptrdiff_t(top));
    a[top] = a[top] ? a[top] - 1 : kP - 1;
    a[0] = 1;
    std::fill(a.begin() + 1, a.begin() + std::ptrdiff_t(half), 0);
}

// Level 1, one-dimensional in x: Q_1 = 1 + y q1 + y^2 q2 with q1 = -2 ge and q2 = v (v at the
// start of level 3's slot, series of length m/2). Keeps the transforms of q1 and q2 at m as the
// level, and with FirstLevelProducts' G and E (after v in level 3's slot), writes
// Q_2 = Q_1(x) Q_1(-x) mod u^(m/4), u = x^2,
//   Q_2 = 1 + 2y q1e + y^2 (2 q2e + G(q1)) + 2y^3 E(q1, q2) + y^4 G(q2),
// at stride m/2 into level 2 (ae: the even part of a, as a series in u).
inline void first_level(const Transform& t, std::span<const std::uint32_t> g, Levels& levels) {
    const std::size_t m = levels.m, h = m / 2, quarter = m / 4;
    const Tables& tables = levels.tables;
    const std::span<const std::uint32_t> v = levels.level(3).first(h);
    const auto sum = [](std::uint32_t x, std::uint32_t y) { return x + y >= kP ? x + y - kP : x + y; };
    const std::span<std::uint32_t> t1 = levels.first_transform(0), t2 = levels.first_transform(1);
    const std::size_t even = std::max<std::size_t>(std::min(h, (g.size() + 1) / 2), 1);  // q1[j] for 0 < j < even
    t1[0] = 0;
    for (std::size_t j = 1; j < even; ++j) t1[j] = minus_twice(g[2 * j]);
    std::fill(t1.begin() + std::ptrdiff_t(even), t1.begin() + std::ptrdiff_t(h), 0);
    const std::span<std::uint32_t> q = levels.level(2);  // Q_2, x below m/4 (the rest is not read)
    for (std::size_t i = 0; i < quarter; ++i) q[h + i] = sum(t1[2 * i], t1[2 * i]);
    t.forward(t1.first(h), 0, t1);
    t.forward(v, 0, t2);
    std::uint32_t* out[3];  // G(q1), G(q2), E(q1, q2)
    for (std::size_t i = 0; i < 3; ++i) out[i] = levels.level(3).data() + Arena::footprint(m) + i * Arena::footprint(h);
    const FirstLevelProducts<false> products{&tables, t1.data(), t2.data(), nullptr, {out[0], out[1], out[2]}};
    for (std::size_t k = 0; k < m / 16; k += 8) products.pairs(k);
    for (std::uint32_t* x : out)
        inverse_with(std::span<std::uint32_t>(x, h), tables, InverseBottom{tables.inverse_roots, x}, graeffe_scale(levels.lg - 1),
                     Half::kLower);
    q[0] = 1;
    std::fill(q.begin() + 1, q.begin() + std::ptrdiff_t(quarter), 0);
    for (std::size_t i = 0; i < quarter; ++i) q[2 * h + i] = sum(sum(v[2 * i], v[2 * i]), out[0][i]);
    for (std::size_t i = 0; i < quarter; ++i) q[3 * h + i] = sum(out[2][i], out[2][i]);
    std::copy_n(out[1], quarter, q.begin() + std::ptrdiff_t(4 * h));
}

// Level 1 of compose, one-dimensional: from P_2 (rows 0 .. 3 at stride m/2 in in[2m, 4m), x below
// m/4), rows 2 and 3 of R = P_2(x^2, y) Q_1(-x, y) (x below m/2) into out[2m, 3m) and out[3m, 4m):
//   R_2 = P2(x^2) + P1(x^2) q1(-x) + P0(x^2) q2(-x),  R_3 = P3(x^2) + P2(x^2) q1(-x) + P1(x^2) q2(-x),
// with Pk row k of P_2. Products of length m from the transforms of P0, P1, P2 at m/2 (in out) and
// of q1, q2 (level 1), by CompositionSumBottom.
inline void first_level_transposed(const Transform& t, const Levels& levels, std::span<const std::uint32_t> in,
                                   std::span<std::uint32_t> out) {
    const std::size_t m = levels.m, h = m / 2, quarter = m / 4;
    const Tables& tables = levels.tables;
    const auto row = [&](std::size_t k) { return in.subspan(2 * m + k * h, quarter); };
    std::span<std::uint32_t> tp[3];
    for (std::size_t k = 0; k < 3; ++k) {
        tp[k] = out.subspan(k * Arena::footprint(h), h);
        t.forward(row(k), 0, tp[k]);
    }
    const std::uint32_t *q1 = levels.first_transform(0).data(), *q2 = levels.first_transform(1).data();
    const auto term = [&](std::size_t k, const std::uint32_t* q) { return CompositionBottom{&tables, tp[k].data(), q}; };
    for (std::size_t r = 2; r < 4; ++r) {
        const std::span<std::uint32_t> result = out.subspan(r * m, m);
        inverse_with(result, tables, CompositionSumBottom{{term(r - 1, q1), term(r - 2, q2)}}, kInverseScales[1][levels.lg],
                     Half::kLower);
        const std::span<const std::uint32_t> add_row = row(r);
        for (std::size_t i = 0; i < quarter; ++i) {
            const std::uint32_t x = result[2 * i] + add_row[i];
            result[2 * i] = x >= kP ? x - kP : x;
        }
    }
}

// out = the coefficients of the sum of a b mod (y^n - 1) over K pairs of transforms of length
// n = out.size() (Transform::inverse_product_sum, which takes up to 3), output half only.
template <std::size_t K>
void inverse_product_sum(const Tables& tables, const Transform::Pair* pairs, std::span<std::uint32_t> out, Half output) {
    InverseProductSumBottom<K> bottom{};
    for (std::size_t k = 0; k < K; ++k) bottom.terms[k] = {{tables.roots, tables.inverse_roots, pairs[k].b.data()}, pairs[k].a.data()};
    inverse_with(out, tables, bottom, kInverseScales[1][std::countr_zero(out.size())], output);
}

// t = c a for transforms (or coefficients) a, canonical.
inline void multiply_constant(std::span<const std::uint32_t> a, std::uint32_t c, std::span<std::uint32_t> t) {
    const Factor f(c);
    for (std::size_t i = 0; i < a.size(); i += 8) store(t.data() + i, reduce(times(load(a.data() + i), f), kP));
}

// Level T - 3, one-dimensional in y: Q_(T-3) = sum_(k < 8) x^k q_k(y) at stride 16 in level T - 3
// (rows 0 .. m/8; q_0 = 1, q_k(0) = 0). Keeps the transforms of length m/4 of c_k = (-1)^k q_k,
// 0 < k < 8, as the level, and writes c'_k = (-1)^k q'_k for Q_(T-2) = Q_(T-3)(x) Q_(T-3)(-x) mod u^4
// = 1 + u q'_1 + u^2 q'_2 + u^3 q'_3 (u = x^2), coefficients 1 .. m/4, into c[k] (k = 1, 2, 3):
//   c'_1 = c1^2 - 2 c2,  c'_2 = c2^2 - 2 c1 c3 + 2 c4,  c'_3 = c3^2 + 2 c1 c5 - 2 c2 c4 - 2 c6.
// The products have degree <= m/4 and no y^0 term: mod (y^(m/4) - 1), their y^(m/4) lands on y^0.
inline void third_last_level(const Transform& t, const Levels& levels, const std::span<std::uint32_t> (&c)[4]) {
    const std::size_t m = levels.m, quarter = m / 4, eighth = m / 8;
    const std::span<const std::uint32_t> q = levels.level(levels.lg - 3);
    const std::span<std::uint32_t> temporary = levels.level(levels.lg - 1);  // filled after this level
    std::span<std::uint32_t> coefficients[8], tc[8];
    for (std::size_t k = 1; k < 8; ++k) {
        coefficients[k] = temporary.subspan((k - 1) * Arena::footprint(eighth + 1), eighth + 1);
        tc[k] = levels.column_transform(k);
    }
    for (std::size_t r = 0; r < eighth; r += 8) {  // rows r .. r + 7, transposed
        Vec x[8];
#pragma GCC unroll 8
        for (std::size_t i = 0; i < 8; ++i) x[i] = load(q.data() + 16 * (r + i));
        transpose(x);
#pragma GCC unroll 7
        for (std::size_t k = 1; k < 8; ++k) store(coefficients[k].data() + r, k % 2 ? negate(x[k]) : x[k]);
    }
    for (std::size_t k = 1; k < 8; ++k) {
        const std::uint32_t x = q[16 * eighth + k];
        coefficients[k][eighth] = k % 2 && x ? kP - x : x;
    }
    for (std::size_t k = 1; k < 8; ++k) t.forward(coefficients[k], 0, tc[k]);  // over q
    std::span<std::uint32_t> twice[3], square[4];  // 2 c5, -2 c3, -2 c4; the product sums
    const std::span<std::uint32_t> sums = temporary.subspan(7 * Arena::footprint(eighth + 1));
    for (std::size_t i = 0; i < 3; ++i) twice[i] = sums.subspan(i * Arena::footprint(quarter), quarter);
    for (std::size_t k = 1; k < 4; ++k) square[k] = sums.subspan((k + 2) * Arena::footprint(quarter), quarter);
    multiply_constant(tc[5], 2, twice[0]), multiply_constant(tc[3], kP - 2, twice[1]), multiply_constant(tc[4], kP - 2, twice[2]);
    const Transform::Pair first[1] = {{tc[1], tc[1]}}, second[2] = {{tc[2], tc[2]}, {tc[1], twice[1]}},
                          third[3] = {{tc[3], tc[3]}, {tc[1], twice[0]}, {tc[2], twice[2]}};
    t.inverse_product_sum(first, square[1]);
    t.inverse_product_sum(second, square[2]);
    t.inverse_product_sum(third, square[3]);
    const std::uint32_t weight[4] = {0, kP - 2, 2, kP - 2};
    for (std::size_t k = 1; k < 4; ++k) {  // c[k][i] = square[k][i mod m/4] + weight[k] c_2k[i]
        const std::span<const std::uint32_t> linear = coefficients[2 * k];
        const Factor w(weight[k]);
        for (std::size_t i = 0; i < quarter; i += 8) {
            const Vec x = load(square[k].data() + i);
            store(c[k].data() + i, i < eighth ? canonical(add(x, reduce(times(load(linear.data() + i), w), kP))) : x);
        }
        const std::uint64_t last = c[k][eighth] + std::uint64_t(weight[k]) * linear[eighth];
        c[k][eighth] = std::uint32_t(last % kP);
        c[k][quarter] = c[k][0];
        c[k][0] = 0;
        std::fill(c[k].begin() + std::ptrdiff_t(quarter + 1), c[k].end(), 0);
    }
}

// Level T - 3 of compose, one-dimensional in y: from P_(T-2) = sum_(j < 4) x^j P_j(y) (p[j]: m/4
// coefficients each), rows m/8 .. m/4 - 1 of R_c = [c even] P_(c/2) + sum_(2j + k = c, k > 0) P_j c_k
// for c < 8 (products mod (y^(m/4) - 1), c_k as in third_last_level) into out at stride 16 from
// out[2m]: P_(T-3) reversed, level T - 4's input. work: 11 spans of m/4 (the transforms of p and
// the R_c). out must not overlap p or work.
inline void third_last_level_transposed(const Transform& t, const Levels& levels, const std::span<const std::uint32_t> (&p)[4],
                                        std::span<std::uint32_t> work, std::span<std::uint32_t> out) {
    const std::size_t m = levels.m, quarter = m / 4, eighth = m / 8;
    std::span<std::uint32_t> tp[4], r[8];
    for (std::size_t j = 0; j < 4; ++j) {
        tp[j] = work.subspan(j * Arena::footprint(quarter), quarter);
        t.forward(p[j], 0, tp[j]);
    }
    for (std::size_t c = 1; c < 8; ++c) {
        r[c] = work.subspan((c + 3) * Arena::footprint(quarter), quarter);
        Transform::Pair pairs[4];
        std::size_t count = 0;
        for (std::size_t j = 0; j < 4 && 2 * j < c; ++j) pairs[count++] = {tp[j], levels.column_transform(c - 2 * j)};
        if (count < 4) t.inverse_product_sum(std::span(pairs, count), r[c], Half::kUpper);
        else inverse_product_sum<4>(levels.tables, pairs, r[c], Half::kUpper);
    }
    for (std::size_t i = eighth; i < quarter; i += 8) {  // rows i .. i + 7 from the columns
        Vec x[8];
        x[0] = load(p[0].data() + i);
#pragma GCC unroll 7
        for (std::size_t c = 1; c < 8; ++c) x[c] = c % 2 ? load(r[c].data() + i) : canonical(add(load(r[c].data() + i), load(p[c / 2].data() + i)));
        transpose(x);
#pragma GCC unroll 8
        for (std::size_t j = 0; j < 8; ++j) store(out.data() + 2 * m + 16 * (i - eighth + j), x[j]);
    }
}

// The transforms of all levels of g. t: lg_max >= levels.lg + 1.
//  - Level 0, Q_0 = 1 - y g: G = T_2m(g); with u = x^2, g(x) g(-x) = v(u) from G by LevelBottom's
//    pairs, and Q_1 = 1 - 2y ge(u) + y^2 v(u) mod u^(m/2), g = ge(x^2) + x go(x^2).
//  - Level 1: first_level.
//  - Levels 2 .. T - 4: Q_s at stride 2L has x below L (vector bit T - s - 3 zero: pruned forward);
//    Q_(s+1) keeps v at stride L for x below L / 2 (bit T - s - 4: pruned inverse).
//  - Level T - 3: third_last_level gives Q_(T-2) = 1 + x q1 + x^2 q2 + x^3 q3, deg q_k <= m/4.
//    Then Q_(T-1) = 1 + x q with q = 2 q2 - q1^2 (deg <= m/2; q1^2 has no y^0 term, so its
//    y^(m/2) wraps onto y^0).
inline void build_levels(const Transform& t, std::span<const std::uint32_t> g, Levels& levels) {
    const std::size_t m = levels.m;
    const std::span<std::uint32_t> v = levels.level(3).first(m), q0 = levels.level(0).first(2 * m);
    std::fill(q0.begin(), q0.end(), 0);
    std::copy(g.begin() + std::ptrdiff_t(std::min<std::size_t>(g.size(), 1)), g.begin() + std::ptrdiff_t(std::min(g.size(), m)),
              q0.begin() + 1);
    const Tables& tables = levels.tables;
    forward_with(q0, m, tables, LevelBottom{&tables, v.data()});
    inverse_with(v, tables, InverseBottom{tables.inverse_roots, v.data()}, graeffe_scale(levels.lg), Half::kBoth);
    first_level(t, g, levels);
    for (int s = 2; s + 3 < levels.lg; ++s) {  // Q_s: rows 0 .. Y at stride 2L; V in level s + 1
        const std::size_t stride = 2 * (m >> s), pad = std::size_t(levels.lg - s - 3);
        const std::span<std::uint32_t> next = levels.level(s + 1).first(2 * m);
        forward_pruned(levels.level(s), ((std::size_t(1) << s) + 1) * stride, pad, tables, LevelBottom{&tables, next.data()});
        inverse_pruned(next, pad - 1, tables, InverseBottom{tables.inverse_roots, next.data()}, graeffe_scale(levels.lg + 1));
        next_level(levels.level(s + 1), stride / 2, std::size_t(2) << s);
    }
    std::span<std::uint32_t> c[4];  // c[k] = (-1)^k q_k of Q_(T-2)
    for (std::size_t k = 1; k < 4; ++k) c[k] = levels.level(levels.lg - 2).subspan((k - 1) * Arena::footprint(m / 2), m / 2);
    third_last_level(t, levels, c);
    const std::span<std::uint32_t> q = levels.level(levels.lg - 1).first(m);  // -q_(T-1)
    std::fill(q.begin(), q.end(), 0);
    for (std::size_t i = 1; i <= m / 4; ++i) q[i] = minus_twice(c[2][i]);
    const std::span<std::uint32_t> square = levels.level(levels.lg - 1).subspan(Arena::footprint(m), m / 2);  // q1^2
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
    return std::countr_zero(detail::compose_length(n)) + 1;
}

// Scratch words for compose() of n coefficients.
inline std::size_t compose_scratch(std::size_t n) {
    if (n <= detail::kComposeBase) return 0;
    return detail::Levels::scratch(detail::compose_length(n));
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
// The first two and the last three levels are one-dimensional:
//  - level T - 1: P = f reversed, R = p(y) (1 - x q(y)): rows y in [m/2, m) of p and -p q;
//  - level T - 2: P = p0(y) + x^2 p1(y), R = P Q_(T-2)(-x) mod x^4 by columns;
//  - level T - 3: third_last_level_transposed;
//  - level 1: first_level_transposed;
//  - level 0: P = p0(x) + y p1(x), R's row 1 = p1(x^2) - p0(x^2) g(-x) = h.
// Level s's transforms are pruned (composition.hpp's levels): P_(s+1) at stride L has x below L / 2
// (vector bit T - s - 4 zero), and of R only x below L is used.
inline void compose(const Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                    std::span<std::uint32_t> h, std::span<std::uint32_t> scratch) {
    const std::size_t n = h.size();
    if (n <= detail::kComposeBase) return detail::compose_direct(f, g, h);
    detail::Levels levels(detail::compose_length(n), scratch);
    detail::build_levels(t, g, levels);
    const std::size_t m = levels.m;
    const int lg = levels.lg;
    // Level T - 1: columns p0 = p[m/2, m), p1 = (-p q)[m/2, m) of P_(T-2), after level T - 1's
    // transform in its slot.
    const std::span<std::uint32_t> last = levels.level(lg - 1);
    const std::span<std::uint32_t> p = last.subspan(Arena::footprint(m), m), pq = last.subspan(2 * Arena::footprint(m), m);
    for (std::size_t i = 0; i < m; ++i) p[i] = m - 1 - i < f.size() ? f[m - 1 - i] : 0;
    t.cyclic_product(p, 0, pq, last.first(m), Half::kUpper);
    // Level T - 2, length m/2: columns r0 = p0, r1 = -p0 q1, r2 = p1 + p0 q2, r3 = -p1 q1 - p0 q3 of
    // P_(T-2), rows [m/4, m/2): r1 and r2 in the lower halves of p and pq, the transforms t0, t1 of
    // p0, p1 and r3 in level T - 1's slot too. Then level T - 3, its work in level T - 2's slot, its
    // output in the upper half of level T - 3's slot.
    {
        const std::size_t h = m / 2, quarter = m / 4;
        const std::span<std::uint32_t> p0 = p.subspan(h), p1 = pq.subspan(h), t0 = last.first(h), t1 = last.subspan(Arena::footprint(h), h);
        const std::span<std::uint32_t> c = levels.level(lg - 2);
        const auto column = [&](std::size_t k) { return c.subspan((k - 1) * Arena::footprint(h), h); };
        const std::span<std::uint32_t> r1 = p.first(h), r2 = pq.first(h), r3 = last.subspan(3 * Arena::footprint(m), h);
        t.forward(p0, 0, t0);
        t.forward(p1, 0, t1);
        t.inverse_product(t0, column(1), r1, Half::kUpper);
        t.inverse_product(t0, column(2), r2, Half::kUpper);
        const Transform::Pair pairs[2] = {{t1, column(1)}, {t0, column(3)}};
        t.inverse_product_sum(pairs, r3, Half::kUpper);
        for (std::size_t i = quarter; i < h; ++i) {
            const std::uint32_t x = p1[i] + r2[i];
            r2[i] = x >= kModulus ? x - kModulus : x;
        }
        const std::span<const std::uint32_t> columns[4] = {p0.subspan(quarter), r1.subspan(quarter), r2.subspan(quarter),
                                                           r3.subspan(quarter)};
        detail::third_last_level_transposed(t, levels, columns, levels.level(lg - 2), levels.level(lg - 3));
    }
    // Level s reads P_(s+1) from the upper half of in and writes R to out, a slot no longer read.
    std::span<std::uint32_t> in = levels.level(lg - 3);
    const std::uint32_t scale = detail::kInverseScales[1][lg + 2];
    for (int s = lg - 4; s >= 2; --s) {
        const std::span<std::uint32_t> x = in.subspan(2 * m), out = levels.level(s == lg - 4 ? lg - 2 : s + 1);
        const std::size_t pad = std::size_t(lg - s - 3);
        detail::forward_pruned(x, 2 * m, pad - 1, levels.tables, detail::ForwardBottom{levels.tables.roots});
        detail::inverse_pruned(out, pad, levels.tables, detail::CompositionBottom{&levels.tables, x.data(), levels.level(s).data()},
                               scale, Half::kUpper);
        in = out;
    }
    detail::first_level_transposed(t, levels, in, levels.level(2));
    in = levels.level(2);
    const std::span<std::uint32_t> out = levels.level(1);
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
