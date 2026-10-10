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

// a = P mod (z^8 - t), c and tc as for graeffe_sums: sum = the even (parity 0) or odd (parity 1)
// part of a(z) c(-z) mod (z^8 - t), as a series in u mod (u^4 - t) (the odd part divided by z):
// sum_k adds (-1)^j a_i c_j over i + j = 2k + parity and (-1)^j a_i tc_j over i + j = 2k + 8 +
// parity (tc[j] for j >= 1 - parity). 8 terms, 4 of them negative (offset by 4 P^2), below 8 P^2.
template <int kParity>
[[gnu::always_inline]] inline void product_sums(const Vec (&a)[8], const Vec (&c)[8], const Vec (&tc)[8],
                                                Vec (&sum)[4]) {
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
        Vec s = _mm256_set1_epi64x(static_cast<long long>(4 * std::uint64_t(kP) * kP));
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
[[gnu::always_inline]] inline void graeffe_leaf(const Vec (&c)[8], const Vec (&tc)[8], Vec (&out)[4]) {
    Vec even[4], odd[4], c2[8], tc2[8];
    graeffe_sums(c, tc, even);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) c2[i] = _mm256_srli_epi64(c[i], 32), tc2[i] = _mm256_srli_epi64(tc[i], 32);
    graeffe_sums(c2, tc2, odd);
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) out[k] = low(reduce_wide({even[k], odd[k]}));
}

template <int kParity>
[[gnu::always_inline]] inline void product_leaf(const Vec (&a)[8], const Vec (&c)[8], const Vec (&tc)[8],
                                                Vec (&out)[4]) {
    Vec even[4], odd[4], a2[8], c2[8], tc2[8];
    product_sums<kParity>(a, c, tc, even);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i)
        a2[i] = _mm256_srli_epi64(a[i], 32), c2[i] = _mm256_srli_epi64(c[i], 32), tc2[i] = _mm256_srli_epi64(tc[i], 32);
    product_sums<kParity>(a2, c2, tc2, odd);
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) out[k] = low(reduce_wide({even[k], odd[k]}));
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

// s = r[k + lane] and its inverse, for 8 pairs from pair k (a multiple of 8).
struct PairWeights {
    Factors s, inverse;
    PairWeights(const Tables& tables, std::size_t k) : s(entries(tables.roots, k)), inverse(entries(tables.inverse_roots, k)) {}
};

// The bottom of the forward transform of Q_s (length 4m): forward butterflies, then per pair of
// leaves 2k, 2k + 1 (moduli z^8 -+ s, s = r[k]) the leaves of V = Q_s(x) Q_s(-x) and of
// W = the odd part of P(x) Q_s(-x), from p, P's transform (canonical leaves): mod (u^4 -+ s),
// u = z^2, A and B by graeffe_leaf and product_leaf; leaf k of the transform of length 2m is
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
        const PairWeights weights(*tables, k);
        Vec va[4], vb[4], wa[4], wb[4];
        leaves<false>(q, p + 16 * k, weights.s, va, wa);
        leaves<true>(q + 8, p + 16 * k + 8, weights.s, vb, wb);
        combine_pairs(va, vb, weights.inverse, v + 8 * k);
        combine_pairs(wa, wb, weights.inverse, w + 8 * k);
    }

    // V and W mod (u^4 - t) for t = s (or -s if kNegative) of the 8 leaves at q and p, times 2^-32.
    template <bool kNegative>
    [[gnu::noinline, gnu::flatten]] static void leaves(const std::uint32_t* q, const std::uint32_t* p, const Factors& s,
                                                       Vec (&v)[4], Vec (&w)[4]) {
        Vec c[8], a[8], tc[8];
        load_leaves(q, c), load_leaves(p, a);
        scaled_leaves<kNegative>(c, s, 2, tc);
        graeffe_leaf(c, tc, v);
        product_leaf<1>(a, c, tc, w);
    }
};

// Level 1's five products for 8 leaf pairs from pair k of the transforms of q1, q2, p1 (length m):
// G(q1), G(q2), E(q1, q2), O(p1, q1), O(p1, q2) (projection_first_levels), leaves k .. k + 7 of
// transforms of length m/2 at out[i] + 8k, times 2^-31.
struct FirstLevelProducts {
    const Tables* tables;
    const std::uint32_t *q1, *q2, *p1;
    std::uint32_t* out[5];

    [[gnu::flatten]] void pairs(std::size_t k) const {
        const PairWeights weights(*tables, k);
        Vec a[5][4], b[5][4];
        leaves<false>(16 * k, weights.s, a);
        leaves<true>(16 * k + 8, weights.s, b);
#pragma GCC unroll 5
        for (int i = 0; i < 5; ++i) combine_pairs(a[i], b[i], weights.inverse, out[i] + 8 * k);
    }

    template <bool kNegative>
    [[gnu::noinline, gnu::flatten]] void leaves(std::size_t at, const Factors& s, Vec (&r)[5][4]) const {
        Vec c1[8], c2[8], a[8], t1[8], t2[8];
        load_leaves(q1 + at, c1), load_leaves(q2 + at, c2), load_leaves(p1 + at, a);
        scaled_leaves<kNegative>(c1, s, 2, t1);
        scaled_leaves<kNegative>(c2, s, 1, t2);
        graeffe_leaf(c1, t1, r[0]);
        graeffe_leaf(c2, t2, r[1]);
        product_leaf<0>(c1, c2, t2, r[2]);
        product_leaf<1>(a, c1, t1, r[3]);
        product_leaf<1>(a, c2, t2, r[4]);
    }
};

// Levels 0 and 1, one-dimensional in x, into q and p (length 4m each), ending in level 2's
// layout (stride m/2: Q_2 rows 0 .. 4, P_2 rows 0 .. 3, x below m/4).
// Level 0: Q_0 = 1 - y g, P_0 = x^e (e = m - n); LevelBottom on the transform of g at 2m gives
// v = g(x) g(-x) as a series in u = x^2. Level 1, series in x of length m/2:
// Q_1 = 1 + y q1 + y^2 q2 with q1 = -2 ge (g = ge(x^2) + x go(x^2)) and q2 = v, and
// P_1 = p0 + y p1 with p0 = x^d if e = 2d + 1 (else 0) and p1[j] = -(-1)^i g_i for i = 2j + 1 - e.
// With G(a) = a(x) a(-x) and E(a, b), O(a, b) the even part and the odd part (over x) of
// a(x) b(-x), all series in u, and ae, ao the even and odd parts of a:
//   Q_2 = 1 + 2y q1e + y^2 (2 q2e + G(q1)) + 2y^3 E(q1, q2) + y^4 G(q2),
//   P_2 = p0o + y (p1o + O(p0, q1)) + y^2 (O(p1, q1) + O(p0, q2)) + y^3 O(p1, q2), mod u^(m/4).
// O(p0, b) is a shift of b. The five products: FirstLevelProducts on the transforms of q1, q2,
// p1 at m, inverses at m/2 (lower half). In units of h = m/2: q holds the transforms of q2 and q1
// (q[0, 4h)), then O(p1, q1), O(p1, q2) at q[5h, 7h); p the transform of p1 (p[0, 2h)), G(q1),
// G(q2), E(q1, q2) at p[4h, 7h) and q2 at p[7h, 8h). P_2 goes to p[0, 4h), then Q_2 to q[0, 5h).
inline void projection_first_levels(const Transform& t, const Tables& tables, std::span<const std::uint32_t> g,
                                    std::size_t n, std::span<std::uint32_t> q, std::span<std::uint32_t> p) {
    using ntt::detail::multiply_mod, ntt::detail::power;
    const std::size_t m = q.size() / 4, h = m / 2, quarter = m / 4, e = m - n, size = std::min(n, g.size());
    const auto at = [&](std::size_t i) { return i > 0 && i < size ? g[i] : 0; };
    const auto negated = [](std::uint32_t x) { return x ? kP - x : 0; };
    const auto sum = [](std::uint32_t x, std::uint32_t y) { return x + y >= kP ? x + y - kP : x + y; };
    const auto q1 = [&](std::size_t j) { return j < h ? minus_twice(at(2 * j)) : 0; };
    const auto p1 = [&](std::size_t j) {  // -(-1)^i g_i
        const std::size_t i = 2 * j + 1 - e;
        return 2 * j + 1 < e ? 0 : i % 2 ? at(i) : negated(at(i));
    };
    // Coefficient i of O(p0, b): (-1)^j b[j] for j = 2i + 1 - d, or 0.
    const auto shifted = [&](auto b, std::size_t i) {
        const std::size_t d = (e - 1) / 2, j = 2 * i + 1 - d;
        if (e % 2 == 0 || 2 * i + 1 < d || j >= h) return 0u;
        return j % 2 ? negated(b(j)) : b(j);
    };

    for (std::size_t i = 0; i < size; ++i) q[i] = at(i);
    forward_with(q.first(2 * m), size, tables, LevelBottom{&tables, q.data()});
    t.inverse(q.first(m));  // v
    const std::span<std::uint32_t> q2 = p.subspan(7 * h, h);
    std::copy_n(q.begin(), h, q2.begin());
    const auto q2_at = [&](std::size_t j) { return q2[j]; };

    const std::span<std::uint32_t> tq2 = q.first(m), tq1 = q.subspan(m, m), tp1 = p.first(m);
    for (std::size_t j = 0; j < h; ++j) tq1[j] = q1(j), tp1[j] = p1(j);
    t.forward(tq2.first(h), 0, tq2);
    t.forward(tq1.first(h), 0, tq1);
    t.forward(tp1.first(h), 0, tp1);
    std::uint32_t* const g1 = p.data() + 4 * h;
    std::uint32_t* const g2 = p.data() + 5 * h;
    std::uint32_t* const e12 = p.data() + 6 * h;
    std::uint32_t* const o1 = q.data() + 5 * h;
    std::uint32_t* const o2 = q.data() + 6 * h;
    const FirstLevelProducts products{&tables, tq1.data(), tq2.data(), tp1.data(), {g1, g2, e12, o1, o2}};
    for (std::size_t k = 0; k < m / 16; k += 8) products.pairs(k);
    const std::uint32_t scale = multiply_mod(power(std::uint32_t(m / 16), kP - 2), std::uint32_t((1ULL << 31) % kP));
    for (std::uint32_t* out : products.out)
        inverse_with(std::span<std::uint32_t>(out, h), tables, InverseBottom{tables.inverse_roots, out}, scale, Half::kLower);

    std::fill(p.begin(), p.begin() + std::ptrdiff_t(4 * h), 0);
    if (e % 2 && (e - 1) / 2 % 2) p[(e - 3) / 4] = 1;
    for (std::size_t i = 0; i < quarter; ++i) {
        p[h + i] = sum(p1(2 * i + 1), shifted(q1, i));
        p[2 * h + i] = sum(o1[i], shifted(q2_at, i));
        p[3 * h + i] = o2[i];
    }
    std::fill(q.begin(), q.begin() + std::ptrdiff_t(5 * h), 0);
    q[0] = 1;
    for (std::size_t i = 0; i < quarter; ++i) {
        const std::uint32_t x = q1(2 * i), y = q2[2 * i];
        q[h + i] = sum(x, x);
        q[2 * h + i] = sum(sum(y, y), g1[i]);
        q[3 * h + i] = sum(e12[i], e12[i]);
        q[4 * h + i] = g2[i];
    }
}

// Levels T - 2 and T - 1, one-dimensional in y, from V in q[0, 2m) and W in p[0, 2m) (level
// T - 3's, at stride 8, m/4 rows, V wrapped: row 0 holds 1 + row m/4):
// Q_(T-2) = 1 + x q1 + x^2 q2 + x^3 q3 and P_(T-2) = p0 + x p1 + x^2 p2 + x^3 p3, deg q_k <= m/4,
// deg p_k < m/4. With c_k = (-1)^k q_k, a = [x^3] P_(T-2) / Q_(T-2) = r3 + r1 (c1^2 - 2 c2) for
// r1 = p1 + p0 c1 and r3 = p3 + p2 c1 + p1 c2 + p0 c3 (products of length m/2), the last product
// of length m. c1^2 has no y^0 term, so its y^(m/2) wraps onto y^0. The columns go to the upper
// halves of q and p, everything after them to the lower halves.
inline void projection_last_levels(const Transform& t, std::span<std::uint32_t> q, std::span<std::uint32_t> p,
                                   std::span<std::uint32_t> a) {
    const std::size_t m = q.size() / 4, h = m / 2, rows = m / 4;
    std::span<std::uint32_t> c[4], pc[4], pt[3];  // c[k], k >= 1: transforms; pc: coefficients; pt: transforms of pc
    for (std::size_t k = 1; k < 4; ++k) {
        c[k] = q.subspan(2 * m + (k - 1) * h, h);
        std::fill(c[k].begin(), c[k].end(), 0);
        for (std::size_t i = 1; i <= rows; ++i) {
            const std::uint32_t x = q[8 * (i % rows) + k];
            c[k][i] = k % 2 && x ? kP - x : x;
        }
    }
    for (std::size_t k = 0; k < 4; ++k) {
        pc[k] = p.subspan(2 * m + k * rows, rows);
        for (std::size_t i = 0; i < rows; ++i) pc[k][i] = p[8 * i + k];
    }
    const std::span<std::uint32_t> negative_q = p.subspan(3 * m, m);  // c1^2 - 2 c2
    std::fill(negative_q.begin(), negative_q.end(), 0);
    for (std::size_t i = 1; i <= rows; ++i) negative_q[i] = minus_twice(c[2][i]);
    for (std::size_t k = 1; k < 4; ++k) t.forward(c[k]);
    for (std::size_t k = 0; k < 3; ++k) {
        pt[k] = q.subspan(k * h, h);
        t.forward(pc[k], 0, pt[k]);
    }
    const std::span<std::uint32_t> square = q.subspan(3 * h, h), r1 = p.first(h), r3 = p.subspan(h, h);
    const std::span<std::uint32_t> product = p.subspan(m, m);
    t.inverse_product(c[1], c[1], square);
    for (std::size_t i = 1; i <= h; ++i) {
        const std::uint32_t x = negative_q[i] + square[i % h];
        negative_q[i] = x >= kP ? x - kP : x;
    }
    t.inverse_product(pt[0], c[1], r1);
    const Transform::Pair pairs[3] = {{pt[2], c[1]}, {pt[1], c[2]}, {pt[0], c[3]}};
    t.inverse_product_sum(pairs, r3);
    for (std::size_t i = 0; i < rows; ++i) {
        const std::uint32_t x = r1[i] + pc[1][i], y = r3[i] + pc[3][i];
        r1[i] = x >= kP ? x - kP : x, r3[i] = y >= kP ? y - kP : y;
    }
    t.forward(negative_q);
    t.cyclic_product(r1, 0, product, negative_q);
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
    return Arena::footprint(detail::Tables::words(m)) + 2 * Carve::words(4 * m);
}

// a[k] = [x^(n-1)] g^k for k < n = a.size() >= 1, g[0] = 0 (if g is not empty); coefficients
// past g.size() are zero. scratch: projection_scratch(n) words, 32-byte aligned (from an Arena).
// t: lg_max >= projection_log(n). a must not overlap g.
//
// Levels as above: 0 and 1 (one-dimensional in x, projection_first_levels), 2 .. T - 3 (Kronecker
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
    projection_first_levels(t, tables, g, n, q, p);
    // Inverses at 2m: undo the factor 2m / 8 and ProjectionBottom's 2^-31.
    const std::uint32_t scale = multiply_mod(power(std::uint32_t(m / 4), kP - 2), std::uint32_t((1ULL << 31) % kP));
    for (int s = 2; s + 2 < lg; ++s) {
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
    projection_last_levels(t, q, p, a);
}

}  // namespace poly
