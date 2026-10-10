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

// g = exp(f) mod x^n, n = g.size() <= kExpBase, from d = f': n g_n = sum_k d[k - 1] g[n - k].
inline void exp_direct(std::span<const std::uint32_t> d, std::span<std::uint32_t> g) {
    using ntt::detail::multiply_mod;
    std::uint32_t inv[kExpBase] = {0, 1};  // inv[i] = 1 / i
    for (std::size_t i = 2; i < g.size(); ++i) inv[i] = multiply_mod(kP - kP / std::uint32_t(i), inv[kP % i]);
    g[0] = 1;
    for (std::size_t i = 1; i < g.size(); ++i) {
        std::uint64_t sum = 0;  // fewer than kExpBase terms < P
        for (std::size_t k = 1; k <= i; ++k) sum += std::uint64_t(d[k - 1]) * g[i - k] % kP;
        g[i] = multiply_mod(std::uint32_t(sum % kP), inv[i]);
    }
}

}  // namespace detail

namespace detail {

// Words of d = f' and of the larger scratch buffers: a power of two >= n, at least 2 kExpBase.
inline std::size_t exp_length(std::size_t n) {
    return std::size_t(1) << std::max(Transform::kMinLog + 1, int(std::bit_width(std::max<std::size_t>(n, 2) - 1)));
}

}  // namespace detail

// Transform length exp uses for n coefficients: the Transform needs lg_max >= this. The last
// step's transforms have length exp_length(n) / 2, as do the full steps' doublings.
inline int exp_log(std::size_t n) { return std::countr_zero(detail::exp_length(n)) - 1; }

namespace detail {

// x - y mod P for canonical x, y.
inline Vec difference(Vec x, Vec y) { return reduce(_mm256_sub_epi32(add(x, broadcast(kP)), y), kP); }

// The last Newton step of exp_newton() below, from g mod x^m to g mod x^n, m < n <= 2m, with
// transforms of length m only. From the previous step: H = T_m(h0), h0 = 1 / g mod x^(m/2), and
// G0 = T_m(g0), g0 = g mod x^(m/2) (computed here when there was no previous step). With the
// halves g1 = g[m/2, m), r0 = r mod x^(m/2), r1 = r[m/2, m), s0 and s1 likewise:
//   h = 1 / g mod x^m = h0 - x^(m/2) (h0 e mod x^(m/2)), e = (g h0)[m/2, m)
//   t = h r mod x^m   = h0 r0 + x^(m/2) (h0 (r1 - e r0) mod x^(m/2))
//   g s mod x^m       = g0 s0 + x^(m/2) ((g0 s1 + g1 s0) mod x^(m/2))
// Each product has degree < m - 1, so a cyclic product of length m gives it exactly; the upper
// parts are needed only if n - m > m/2. (g s)[m/2, m) is the upper half of the cyclic product
// g s0 + g0 x^(m/2) s1 (no product reaches x^(3m/2)). 14 transforms and 8 leaf products of
// length m (a full step, as in exp_newton: 16 and 7).
// Spans: gt = [G0, T_m(g)] and w of 2m words, work of m words; ht holds H.
[[gnu::always_inline]] inline void exp_last_step(const Transform& t, std::span<const std::uint32_t> d, std::span<std::uint32_t> g,
                                                 std::size_t m, bool first, Vec d_before, std::span<std::uint32_t> gt,
                                                 std::span<const std::uint32_t> ht, std::span<std::uint32_t> w,
                                                 std::span<std::uint32_t> work) {
    const std::size_t n = g.size(), rest = n - m, half = m / 2;
    const bool upper = rest > half;
    const Half both = upper ? Half::kBoth : Half::kLower;
    const std::span<std::uint32_t> g0t = gt.first(m), glt = gt.subspan(m, m), r = w.first(m), r0t = w.subspan(m, m);
    if (first) t.forward(g.first(half), 0, g0t);
    t.forward(g.first(m), 0, glt);
    if (upper) {  // T_m(e) in work
        t.inverse_product(glt, ht, r, Half::kUpper);
        t.forward(r.subspan(half, half), 0, work);
    }
    t.forward_product(d.first(m - 1), 1, r, glt);
    t.inverse(r, both);
    t.forward(r.first(half), 0, r0t);
    if (upper) {  // h0 (r1 - e r0) mod x^(m/2) at work[0, m/2)
        t.inverse_product(work, r0t, work, Half::kLower);
        for (std::size_t i = 0; i < half; i += 8) store(work.data() + i, difference(load(r.data() + half + i), load(work.data() + i)));
        t.forward(work.first(half), 0, work);
        t.inverse_product(ht, work, work, Half::kLower);
    }
    t.inverse_product(ht, r0t, r0t, both);  // h0 r0
    // s at w[0, rest), from t = h0 r0 + x^(m/2) work[0, m/2) (as in exp_newton)
    detail::divide_by_index(m, w.first(rest), [&](std::size_t i) {
        Vec x = load(r0t.data() + i);
        if (i >= half) x = reduce(add(x, load(work.data() + i - half)), kP);
        x = reduce(add(x, load_unaligned(d.data() + m - 1 + i)), kP);
        return add(x, _mm256_sub_epi32(broadcast(kP), i ? load_unaligned(d.data() + i - 1) : d_before));
    });
    t.forward(w.first(std::min(rest, half)), 0, work);  // T_m(s0)
    if (upper) {  // (g s)[m/2, rest) at r0t[m/2, ..)
        t.forward(w.subspan(half, rest - half), half, r0t);
        const Transform::Pair pairs[] = {{glt, work}, {g0t, r0t}};
        t.inverse_product_sum(pairs, r0t, Half::kUpper);
        std::copy(r0t.begin() + std::ptrdiff_t(half), r0t.begin() + std::ptrdiff_t(rest), g.begin() + std::ptrdiff_t(m + half));
    }
    t.inverse_product(g0t, work, work, Half::kLower);  // (g s)[0, m/2) = (g0 s0)[0, m/2)
    std::copy_n(work.begin(), std::min(rest, half), g.begin() + std::ptrdiff_t(m));
}

// The Newton steps of exp() below, from g mod x^kExpBase (given) to g mod x^n, n = g.size() >
// kExpBase, for g' = d g: d has exp_length(n) words, d[i] = 0 for i >= n - 1. Each step is
// invariant under scaling g, so g[0] may be any nonzero constant. scratch: exp_newton_scratch(n).
[[gnu::always_inline]] inline void exp_newton(const Transform& t, std::span<const std::uint32_t> d, std::span<std::uint32_t> g,
                                              std::span<std::uint32_t> scratch) {
    const std::size_t n = g.size(), len = d.size();
    const auto take = [&scratch](std::size_t words) {
        const std::span<std::uint32_t> s = scratch.first(words);
        scratch = scratch.subspan(Arena::footprint(words));
        return s;
    };
    const std::span<std::uint32_t> h = take(len / 2), gt = take(len), ht = take(len / 2), w = take(len);
    std::size_t m = kExpBase;
    inverse_direct(g, h.first(m / 2));
    t.forward(h.first(m / 2), 0, ht.first(m));
    // d[i - 1] for i = 0 .. 7, with d[-1] = 0
    const Vec d_before = _mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(load(d.data()), _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6)), _mm256_setzero_si256(), 1);
    for (; 2 * m < n; m *= 2) {
        const std::size_t half = m / 2;
        const std::span<std::uint32_t> g_low = gt.first(m), h_low = ht.first(m), w_low = w.first(m);
        t.forward(g.first(m), 0, g_low);
        // h[m/2, m) = -(h e mod x^(m/2)), e = (g h)[m/2, m)
        t.inverse_product(g_low, h_low, w_low, Half::kUpper);
        t.cyclic_product(w_low.subspan(half), half, w_low, h_low, Half::kUpper, kP - 1);
        std::copy_n(w.begin() + std::ptrdiff_t(half), half, h.begin() + std::ptrdiff_t(half));
        // T_m(r) = T_m(x q) G_lo at w[0, m), r at gt[m, 2m) until G's upper half goes there
        t.forward_product(d.first(m - 1), 1, w_low, g_low);
        t.inverse(w_low, gt.subspan(m, m));
        // t = h r mod x^m from T_2m(r) = [T_m(r), its upper half] and T_2m(h)
        t.forward(h.first(m), 0, ht.first(2 * m));
        t.forward_upper(gt.subspan(m, m), 0, w.subspan(m, m));
        t.inverse_product(w.first(2 * m), ht.first(2 * m), w.first(2 * m), Half::kLower);
        // s at w[m, 2m), then g s mod x^m there
        detail::divide_by_index(m, w.subspan(m, m), [&](std::size_t i) {
            const Vec x = reduce(add(load(w.data() + i), load_unaligned(d.data() + m - 1 + i)), kP);
            return add(x, _mm256_sub_epi32(broadcast(kP), i ? load_unaligned(d.data() + i - 1) : d_before));
        });
        t.forward_upper(g.first(m), 0, gt.subspan(m, m));
        t.cyclic_product(w.subspan(m, m), m, w.first(2 * m), gt.first(2 * m), Half::kUpper);
        std::copy_n(w.begin() + m, m, g.begin() + m);
    }
    exp_last_step(t, d, g, m, m == kExpBase, d_before, gt, ht, w, h);
}

inline std::size_t exp_newton_scratch(std::size_t n) {
    const std::size_t len = exp_length(n);
    return 2 * Arena::footprint(len) + 2 * Arena::footprint(len / 2);
}

}  // namespace detail

// Scratch words for exp() of n coefficients.
inline std::size_t exp_scratch(std::size_t n) {
    return Arena::footprint(detail::exp_length(n)) + detail::exp_newton_scratch(n);
}

// g = exp(f) mod x^n for n = g.size() >= 1. f[0] = 0; coefficients of f past f.size() are zero.
// g may be f (f is read first). scratch: exp_scratch(n) words, 32-byte aligned (from an Arena).
// t: lg_max >= exp_log(n).
//
// Newton steps double the known prefix g = exp(f) mod x^m, keeping h = 1 / g mod x^(m/2) and
// its transform of length m. With d = f' and q = d mod x^(m-1), by transforms of length m and 2m:
//   h = h - x^(m/2) (h (g h)[m/2, m) mod x^(m/2))   now h = 1 / g mod x^m
//   r = x q g mod (x^m - 1)                        = (g q - g') / x^(m-1) + x g'
//   t = h r mod x^m                                = h (g q - g') / x^(m-1) + x q mod x^m
//   s = (f - log g)[m, 2m) = (d[m-1, 2m-1) + t - x q) / (m + i)
//   g[m, 2m) = g s mod x^m
// since g'/g = q - h (g q - g') mod x^(2m-1). T_m(r) = T_m(x q) T_m(g) is the lower half of
// T_2m(r). Per step: 8 transforms of length 2m and 3.5 leaf products of that length.
inline void exp(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size(), len = exp_length(n);
    const std::span<std::uint32_t> d = scratch.first(len);
    derivative(f.first(std::min(f.size(), n)), d);  // d[i] = 0 for i >= n - 1
    exp_direct(d, g.first(std::min(n, kExpBase)));
    if (n > kExpBase) exp_newton(t, d, g, scratch.subspan(Arena::footprint(len)));
}

}  // namespace poly
