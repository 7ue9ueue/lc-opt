// Logarithm of a power series modulo 998244353: g = log(f) mod x^n, f[0] = 1.
// Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::log_log(n)) + poly::log_scratch(n) + ...);
//   poly::Transform t(arena, poly::log_log(n));
//   poly::log(t, f, g, arena.take(poly::log_scratch(n)));  // g.size() == n; f and g do not overlap
#pragma once

#include <immintrin.h>

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

// Up to this many coefficients the logarithm is computed directly.
inline constexpr std::size_t kLogBase = 64;

// g = log(f) mod x^n, n = g.size() <= kLogBase, by i g_i = i f_i - sum_(0<k<i) k g_k f_(i-k).
inline void log_direct(std::span<const std::uint32_t> f, std::span<std::uint32_t> g) {
    using ntt::detail::multiply_mod;
    const auto coefficient = [f](std::size_t i) { return i < f.size() ? f[i] : 0; };
    std::uint32_t inv[kLogBase] = {0, 1}, kg[kLogBase] = {};  // inv[i] = 1 / i, kg[k] = k g_k
    for (std::size_t i = 2; i < g.size(); ++i) inv[i] = multiply_mod(kP - kP / std::uint32_t(i), inv[kP % i]);
    g[0] = 0;
    for (std::size_t i = 1; i < g.size(); ++i) {
        std::uint64_t sum = multiply_mod(std::uint32_t(i), coefficient(i));  // fewer than kLogBase terms < P
        for (std::size_t k = 1; k < i; ++k) sum += kP - std::uint64_t(kg[k]) * coefficient(i - k) % kP;
        kg[i] = std::uint32_t(sum % kP);
        g[i] = multiply_mod(kg[i], inv[i]);
    }
}

}  // namespace detail

// Transform length log uses for n coefficients: the Transform needs lg_max >= this.
inline int log_log(std::size_t n) {
    return std::max(Transform::kMinLog, int(std::bit_width(std::max<std::size_t>(n, 3) - 2)));
}

// Scratch words for log() of n coefficients.
inline std::size_t log_scratch(std::size_t n) { return 3 * Arena::footprint(std::size_t(1) << log_log(n)); }

// g = log(f) mod x^n for n = g.size() >= 1. f[0] = 1; coefficients of f past f.size() are zero.
// f and g must not overlap. scratch: log_scratch(n) words, 32-byte aligned (from an Arena).
// t: lg_max >= log_log(n).
//
// log f is the integral of q = f'/f mod x^(n-1). With transforms of length 2m >= n - 1, d = f'
// and h = 1 / f mod x^m (Karp and Markstein: the division replaces the inverse's last step):
//   q0 = d h mod x^m                    = q mod x^m
//   e = (f q0 - d)[m, 2m)               f q0 = d mod x^m
//   q[m, 2m) = -(h e mod x^m)
// 8 transforms of length 2m and 3 leaf products, after the inverse to m (5 and 2 more).
inline void log(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size();
    if (n <= kLogBase) return log_direct(f, g);
    const std::size_t len = std::size_t(1) << log_log(n), m = len / 2, rest = n - 1 - m;  // 0 < rest <= m
    const std::span<std::uint32_t> ht = scratch.first(len), ft = scratch.subspan(Arena::footprint(len), len),
                                   w = scratch.subspan(2 * Arena::footprint(len), len);
    // d = f' at g[1, n) until q replaces it; g[1 + i] = q[i] / (1 + i).
    const std::span<std::uint32_t> d = g.subspan(1);
    derivative(f.first(std::min(f.size(), n)), d);
    inverse(t, f, ht.first(m), scratch.subspan(Arena::footprint(len), inverse_scratch(m)));  // ft and w
    t.forward(ht.first(m), 0, ht);
    t.cyclic_product(d.first(m), 0, w, ht, Half::kLower);
    detail::divide_by_index(1, g.subspan(1, m), [&w](std::size_t i) { return load(w.data() + i); });
    t.forward(f.first(std::min(f.size(), len)), 0, ft);
    t.cyclic_product(w.first(m), 0, w, ft, Half::kUpper);
    // e at w[m, m + rest), from d[m, n - 1) = g[m + 1, n)
    const std::size_t full = rest / 8 * 8;
    for (std::size_t i = 0; i < full; i += 8)
        store(w.data() + m + i, reduce(_mm256_sub_epi32(add(load(w.data() + m + i), broadcast(kP)),
                                                        load_unaligned(g.data() + m + 1 + i)), kP));
    for (std::size_t i = full; i < rest; ++i) w[m + i] = (w[m + i] + kP - g[m + 1 + i]) % kP;
    t.cyclic_product(w.subspan(m, rest), m, w, ht, Half::kUpper);
    detail::divide_by_index(m + 1, g.subspan(m + 1, rest),
                            [&w, m](std::size_t i) { return _mm256_sub_epi32(broadcast(kP), load(w.data() + m + i)); });
    g[0] = 0;
}

}  // namespace poly
