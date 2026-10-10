// Exponential of a power series modulo 998244353: g = exp(f) mod x^n, f[0] = 0.
// Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::exp_log(n)) + poly::exp_scratch(n) + ...);
//   poly::Transform t(arena, poly::exp_log(n));
//   poly::exp(t, f, g, arena.take(poly::exp_scratch(n)));  // g.size() == n; g may be f
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// Coefficients computed directly before the Newton steps; the first step uses transforms of
// length kExpBase.
inline constexpr std::size_t kExpBase = 64;

// g = exp(f) mod x^n, n = g.size() <= kExpBase, from q = x f': n g_n = sum_k q[k] g[n - k].
inline void exp_direct(std::span<const std::uint32_t> q, std::span<std::uint32_t> g) {
    using ntt::detail::multiply_mod;
    std::uint32_t inv[kExpBase] = {0, 1};  // inv[i] = 1 / i
    for (std::size_t i = 2; i < g.size(); ++i) inv[i] = multiply_mod(kP - kP / std::uint32_t(i), inv[kP % i]);
    g[0] = 1;
    for (std::size_t i = 1; i < g.size(); ++i) {
        std::uint64_t sum = 0;  // fewer than kExpBase terms < P
        for (std::size_t k = 1; k <= i; ++k) sum += std::uint64_t(q[k]) * g[i - k] % kP;
        g[i] = multiply_mod(std::uint32_t(sum % kP), inv[i]);
    }
}

// out[i] = (first + i) a[i] for i < out.size() = a.size(); out may be a (in place).
inline void multiply_by_index(std::span<const std::uint32_t> a, std::size_t first, std::span<std::uint32_t> out) {
    const std::size_t n = out.size(), full = n / 8 * 8;
    Indices index(first);
    for (std::size_t i = 0; i < full; i += 8, index.next())
        store_unaligned(out.data() + i, canonical(montgomery(load_unaligned(a.data() + i), index.value())));
    for (std::size_t i = full; i < n; ++i) out[i] = ntt::detail::multiply_mod(std::uint32_t((first + i) % kP), a[i]);
}

// Words of the larger scratch buffers: a power of two >= n, at least 2 kExpBase.
inline std::size_t exp_length(std::size_t n) {
    return std::size_t(1) << std::max(Transform::kMinLog + 1, int(std::bit_width(std::max<std::size_t>(n, 2) - 1)));
}

}  // namespace detail

// Transform length exp uses for n coefficients: the Transform needs lg_max >= this. The last
// step's transforms have length exp_length(n) / 2, as do the full steps' doublings.
inline int exp_log(std::size_t n) { return std::countr_zero(detail::exp_length(n)) - 1; }

namespace detail {

// A product as Transform's run() computes it (the forward top level of in if the bottom has one,
// the subtrees, the inverse top level times scale), its output half written to `to` (at most n/2
// words, any alignment) instead of out's: the caller needs no copy. out (n words) is the work span.
template <class Bottom>
void product_to(const Transform& t, std::span<std::uint32_t> out, const Source& in, const Bottom& bottom,
                std::uint32_t scale, Half output, std::span<std::uint32_t> to) {
    const std::size_t nv = out.size() / 8;
    auto* v = reinterpret_cast<Vec*>(out.data());
    const Recursion recursion(t.roots(), t.inverse_roots(), bottom);
    const Factor s(scale);
    const auto put = [to, &s](std::size_t j, Vec x) {  // vector j of the half; lanes past to.size() dropped
        const Vec y = reduce(times(x, s), kP);
        if (8 * j + 8 <= to.size()) return store_unaligned(to.data() + 8 * j, y);
        if (8 * j >= to.size()) return;
        const Vec lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const Vec mask = _mm256_cmpgt_epi32(broadcast(std::uint32_t(to.size() - 8 * j)), lanes);
        _mm256_maskstore_epi32(reinterpret_cast<int*>(to.data() + 8 * j), mask, y);
    };
    const bool lower = output == Half::kLower;
    if (std::countr_zero(nv) % 2 == 0) {
        const std::size_t h = nv / 4;
        if constexpr (Bottom::kForward) forward_top4(in, v, h, t.roots());
        for (std::size_t q = 0; q < 4; ++q) recursion.visit(out.data() + 8 * q * h, h, q);
        const Factor z(t.inverse_roots()[1], t.inverse_roots()[9]);
        for (std::size_t j = 0; j < h; ++j) {
            const Vec ab = low(add(v[j], v[j + h])), cd = low(add(v[j + 2 * h], v[j + 3 * h]));
            const Vec amb = low(diff(v[j], v[j + h])), cmd = times(diff(v[j + 2 * h], v[j + 3 * h]), z);
            put(j, lower ? add(ab, cd) : diff(ab, cd));
            put(j + h, lower ? add(amb, cmd) : diff(amb, cmd));
        }
    } else {
        const std::size_t h = nv / 2;
        radix2_halves(recursion, in, out.data(), nv, t.roots());
        for (std::size_t j = 0; j < h; ++j) put(j, lower ? add(v[j], v[j + h]) : diff(v[j], v[j + h]));
    }
}

// The last Newton step of exp_newton() below, from g mod x^m to g mod x^n, m < n <= 2m, with
// transforms of length m only, in blocks of B = m/2 coefficients: g_k = g[kB, kB + B), q_k
// likewise (q = x f'). From the previous step: H = T_m(h0), h0 = 1 / g mod x^B, and G0 = T_m(g0)
// (computed here when there was no previous step). For G = g mod x^(kB), k >= 1,
// g = G exp(f - log G), so g_k = g0 (f - log G)[kB, kB + B) mod x^B. As x (f - log G)' =
// (q G - x G') / G and x G' = q G mod x^(kB), the coefficients of x (f - log G)' from kB on are
// those of (q G) h0:
//   t = ((q mod x^(kB)) G)[kB, kB + B) h0 mod x^B + q_k   (q_k G h0 = q_k mod x^B)
//   g_k = g0 (t / (kB + i)) mod x^B.
// Block 2 from r = (q mod x^m)(g mod x^m) mod (x^m - 1) = x G' + (q G)[m, 2m), G = g mod x^m:
//   t = r[0, B) h0 mod x^B - q_0 + q_2   (x G' h0 = q mod x^B).
// Block 3: (q G)[3B, 4B), G = g mod x^(3B), is (q_1 g_1)[B, 2B) = r[B, 2B) - x G'[B, 2B) plus
//   the upper half of ((q_0 + x^B q_1) g_2 + q_2 (g_0 + x^B g_1)) mod (x^m - 1).
// 14 transforms and 7 leaf products of length m (a full step: 16 and 7); 7 and 3 when n - m <= B.
// q as for exp_newton: q mod x^m in lo (overwritten here), the rest in g[m, n) until g_2 and g_3
// replace it. Spans: gt = [G0, T_m(g)] and w of 2m words, work of m words; ht holds H.
[[gnu::always_inline]] inline void exp_last_step(const Transform& t, std::span<std::uint32_t> lo, std::span<std::uint32_t> g,
                                                 std::size_t m, bool first, std::span<std::uint32_t> gt,
                                                 std::span<const std::uint32_t> ht, std::span<std::uint32_t> w,
                                                 std::span<std::uint32_t> work) {
    const std::size_t n = g.size(), half = m / 2, rest = n - m, count = std::min(rest, half);
    const bool upper = rest > half;  // block 3
    const std::span<std::uint32_t> g0t = gt.first(m), glt = gt.subspan(m, m), qt = w.first(m), r = w.subspan(m, m);
    const auto q = [g, n](std::size_t i) {  // q[i, i + 8) for m <= i < n, from g; zero from n on
        if (i + 8 <= n) return load_unaligned(g.data() + i);
        alignas(32) std::uint32_t x[8] = {};
        std::copy(g.begin() + std::ptrdiff_t(i), g.end(), x);
        return load(x);
    };
    if (first) t.forward(g.first(half), 0, g0t);
    t.forward(g.first(m), 0, glt);
    if (upper) {  // T_m(q mod x^m) in qt, again for block 3
        t.forward(lo, 0, qt);
        t.inverse_product(qt, glt, r);
    } else {
        t.cyclic_product(lo, 0, r, glt, Half::kLower);
    }
    // Block 2: s at r[0, count), then g_2.
    t.forward(r.first(half), 0, work);
    t.inverse_product(ht, work, work, Half::kLower);
    detail::divide_by_index(m, r.first(count), [&](std::size_t i) {
        const Vec x = reduce(add(load(work.data() + i), q(m + i)), kP);
        return add(x, _mm256_sub_epi32(broadcast(kP), load(lo.data() + i)));
    });
    if (upper) t.forward(g.subspan(m, half), 0, lo);  // T_m(q_2): q mod x^m is no longer read
    t.forward(r.first(count), 0, work);
    const std::uint32_t scale = kInverseScales[1][std::countr_zero(m)];
    const Source none(nullptr, 0, 0);
    product_to(t, work, none, InverseProductBottom{{t.roots(), t.inverse_roots(), work.data()}, g0t.data()}, scale,
               Half::kLower, g.subspan(m, count));
    if (!upper) return;
    // Block 3: (q G)[3B, 4B) at r[B, 2B), s at r[0, rest - B), then g_3.
    t.forward(g.subspan(m, half), 0, work);
    const Transform::Pair pairs[] = {{work, qt}, {lo, glt}};
    t.inverse_product_sum(pairs, work, Half::kUpper);
    Indices index(half);
    for (std::size_t i = half; i < m; i += 8, index.next()) {
        const Vec xg = montgomery(load_unaligned(g.data() + i), index.value());  // x G' < 2P
        const Vec sum = add(load(r.data() + i), load(work.data() + i));
        store(r.data() + i, canonical(_mm256_sub_epi32(add(sum, broadcast(2 * kP)), xg)));
    }
    t.forward(r.subspan(half, half), 0, qt);
    t.inverse_product(ht, qt, qt, Half::kLower);
    detail::divide_by_index(m + half, r.first(rest - half),
                            [&](std::size_t i) { return add(load(qt.data() + i), q(m + half + i)); });
    t.forward(r.first(rest - half), 0, qt);
    product_to(t, qt, none, InverseProductBottom{{t.roots(), t.inverse_roots(), qt.data()}, g0t.data()}, scale,
               Half::kLower, g.subspan(m + half, rest - half));
}

// The Newton steps of exp() below, from g mod x^kExpBase (given) to g mod x^n, n = g.size() >
// kExpBase, for x g' = q g: q[i] is lo[i] below m = lo.size() = exp_length(n) / 2 and g[i] from
// m on (g's coefficients replace it); lo is overwritten. Each step is invariant under scaling g,
// so g[0] may be any nonzero constant. scratch: exp_newton_scratch(n).
[[gnu::always_inline]] inline void exp_newton(const Transform& t, std::span<std::uint32_t> lo, std::span<std::uint32_t> g,
                                              std::span<std::uint32_t> scratch) {
    const std::size_t n = g.size(), len = 2 * lo.size();
    const auto take = [&scratch](std::size_t words) {
        const std::span<std::uint32_t> s = scratch.first(words);
        scratch = scratch.subspan(Arena::footprint(words));
        return s;
    };
    const std::span<std::uint32_t> h = take(len / 2), gt = take(len), ht = take(len / 2), w = take(len);
    std::size_t m = kExpBase;
    inverse_direct(g, h.first(m / 2));
    t.forward(h.first(m / 2), 0, ht.first(m));
    for (; 2 * m < n; m *= 2) {
        const std::size_t half = m / 2;
        const std::span<std::uint32_t> g_low = gt.first(m), h_low = ht.first(m), w_low = w.first(m);
        t.forward(g.first(m), 0, g_low);
        // h[m/2, m) = -(h e mod x^(m/2)), e = (g h)[m/2, m)
        t.inverse_product(g_low, h_low, w_low, Half::kUpper);
        const std::uint32_t negated = ntt::detail::multiply_mod(kInverseScales[1][std::countr_zero(m)], kP - 1);
        product_to(t, w_low, Source(w_low.data() + half, half, half), ProductBottom{t.roots(), t.inverse_roots(), h_low.data()},
                   negated, Half::kUpper, h.subspan(half, half));
        // T_m(r) = T_m(q mod x^m) G_lo at w[0, m), r at gt[m, 2m) until G's upper half goes there
        t.forward_product(lo.first(m), 0, w_low, g_low);
        t.inverse(w_low, gt.subspan(m, m));
        // t = h r mod x^m from T_2m(r) = [T_m(r), its upper half] and T_2m(h)
        t.forward(h.first(m), 0, ht.first(2 * m));
        t.forward_upper(gt.subspan(m, m), 0, w.subspan(m, m));
        t.inverse_product(w.first(2 * m), ht.first(2 * m), w.first(2 * m), Half::kLower);
        // s at w[m, 2m), then g s mod x^m there
        detail::divide_by_index(m, w.subspan(m, m), [&](std::size_t i) {
            const Vec x = reduce(add(load(w.data() + i), load(lo.data() + m + i)), kP);
            return add(x, _mm256_sub_epi32(broadcast(kP), load(lo.data() + i)));
        });
        t.forward_upper(g.first(m), 0, gt.subspan(m, m));
        product_to(t, w.first(2 * m), Source(w.data() + m, m, m), ProductBottom{t.roots(), t.inverse_roots(), gt.data()},
                   kInverseScales[1][std::countr_zero(2 * m)], Half::kUpper, g.subspan(m, m));
    }
    exp_last_step(t, lo, g, m, m == kExpBase, gt, ht, w, h);
}

inline std::size_t exp_newton_scratch(std::size_t n) {
    const std::size_t len = exp_length(n);
    return 2 * Arena::footprint(len) + 2 * Arena::footprint(len / 2);
}

}  // namespace detail

// Scratch words for exp() of n coefficients.
inline std::size_t exp_scratch(std::size_t n) {
    return Arena::footprint(detail::exp_length(n) / 2) + detail::exp_newton_scratch(n);
}

// g = exp(f) mod x^n for n = g.size() >= 1. f[0] = 0; coefficients of f past f.size() are zero.
// g may be f (f is read first). scratch: exp_scratch(n) words, 32-byte aligned (from an Arena).
// t: lg_max >= exp_log(n).
//
// Newton steps double the known prefix g = exp(f) mod x^m, keeping h = 1 / g mod x^(m/2) and
// its transform of length m. With q = x f' (so x g' = q g) and Q = q mod x^m, by transforms of
// length m and 2m:
//   h = h - x^(m/2) (h (g h)[m/2, m) mod x^(m/2))   now h = 1 / g mod x^m
//   r = Q g mod (x^m - 1)                          = (Q g)[m, 2m) + x g'
//   t = h r mod x^m                                = h (Q g)[m, 2m) + Q
//   s = (f - log g)[m, 2m) = (q[m, 2m) + t - Q) / (m + i)
//   g[m, 2m) = g s mod x^m
// since x g'/g = Q - x^m h (Q g)[m, 2m) mod x^(2m). T_m(r) = T_m(Q) T_m(g) is the lower half of
// T_2m(r). Per step: 8 transforms of length 2m and 3.5 leaf products of that length. The last
// step works in blocks of m/2 (detail::exp_last_step). q is kept below exp_length(n) / 2 in the
// scratch and from there on in g, which the last step fills only after reading it.
inline void exp(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size(), m = exp_length(n) / 2, size = std::min(f.size(), n), below = std::min(size, m);
    const std::span<std::uint32_t> lo = scratch.first(m);
    multiply_by_index(f.first(below), 0, lo.first(below));
    std::fill(lo.begin() + std::ptrdiff_t(below), lo.end(), 0);
    if (size > m) multiply_by_index(f.subspan(m, size - m), m, g.subspan(m, size - m));  // in place if g is f
    if (n > std::max(size, m)) std::fill(g.begin() + std::ptrdiff_t(std::max(size, m)), g.end(), 0);
    exp_direct(lo, g.first(std::min(n, kExpBase)));
    if (n > kExpBase) exp_newton(t, lo, g, scratch.subspan(Arena::footprint(m)));
}

}  // namespace poly
