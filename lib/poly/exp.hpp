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

// out[i] = -a[i] mod P, a.size() a multiple of 8, both 32-byte aligned.
inline void negate(std::span<const std::uint32_t> a, std::span<std::uint32_t> out) {
    for (std::size_t i = 0; i < a.size(); i += 8) store(out.data() + i, reduce(_mm256_sub_epi32(broadcast(kP), load(a.data() + i)), kP));
}

}  // namespace detail

// Transform length exp uses for n coefficients: the Transform needs lg_max >= this.
inline int exp_log(std::size_t n) {
    return std::max(Transform::kMinLog + 1, int(std::bit_width(std::max<std::size_t>(n, 2) - 1)));
}

// Scratch words for exp() of n coefficients.
inline std::size_t exp_scratch(std::size_t n) {
    const std::size_t len = std::size_t(1) << exp_log(n);
    return 4 * Arena::footprint(len) + Arena::footprint(len / 2);
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
    const std::size_t n = g.size(), len = std::size_t(1) << exp_log(n);
    const auto take = [&scratch](std::size_t words) {
        const std::span<std::uint32_t> s = scratch.first(words);
        scratch = scratch.subspan(Arena::footprint(words));
        return s;
    };
    const std::span<std::uint32_t> d = take(len), h = take(len / 2), gt = take(len), ht = take(len), w = take(len);
    derivative(f.first(std::min(f.size(), n)), d);  // d[i] = 0 for i >= n - 1
    exp_direct(d, g.first(std::min(n, kExpBase)));
    if (n <= kExpBase) return;

    std::size_t m = kExpBase;
    inverse_direct(g, h.first(m / 2));
    t.forward(h.first(m / 2), 0, ht.first(m));
    // d[i - 1] for i = 0 .. 7, with d[-1] = 0
    const Vec d_before = _mm256_blend_epi32(
        _mm256_permutevar8x32_epi32(load(d.data()), _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6)), _mm256_setzero_si256(), 1);
    for (; m < n; m *= 2) {
        const std::size_t half = m / 2;
        const std::span<std::uint32_t> g_low = gt.first(m), h_low = ht.first(m), w_low = w.first(m);
        t.forward(g.first(m), 0, g_low);
        // h[m/2, m) = -(h e mod x^(m/2)), e = (g h)[m/2, m)
        t.inverse_product(g_low, h_low, w_low, Half::kUpper);
        t.cyclic_product(w_low.subspan(half), half, w_low, h_low, Half::kUpper);
        negate(w_low.subspan(half), h.subspan(half, half));
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
        std::copy_n(w.begin() + m, std::min(m, n - m), g.begin() + m);
    }
}

}  // namespace poly
