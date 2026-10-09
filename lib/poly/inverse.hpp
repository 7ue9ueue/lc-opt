// Inverse of a power series modulo 998244353: g = 1 / f mod x^n. Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::inverse_log(n)) + poly::inverse_scratch(n) + ...);
//   poly::Transform t(arena, poly::inverse_log(n));
//   poly::inverse(t, f, g, arena.take(poly::inverse_scratch(n)));  // g.size() == n
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// Below this many coefficients the inverse is computed directly.
inline constexpr std::size_t kInverseBase = 32;

// g = 1 / f mod x^n, n = g.size() <= kInverseBase, by the recurrence f0 g_i = -sum f_j g_(i-j).
inline void inverse_direct(std::span<const std::uint32_t> f, std::span<std::uint32_t> g) {
    using ntt::detail::multiply_mod;
    const std::uint32_t inv0 = ntt::detail::power(f[0], kP - 2);
    g[0] = inv0;
    for (std::size_t i = 1; i < g.size(); ++i) {
        std::uint64_t sum = 0;  // fewer than kInverseBase terms < P
        for (std::size_t j = 1; j <= std::min(i, f.size() - 1); ++j) {
            sum += std::uint64_t(f[j]) * g[i - j] % kP;
        }
        g[i] = multiply_mod(std::uint32_t(sum % kP ? kP - sum % kP : 0), inv0);
    }
}

}  // namespace detail

// Transform length the inverse uses for n coefficients: the Transform needs lg_max >= this.
inline int inverse_log(std::size_t n) {
    return std::max(Transform::kMinLog, int(std::bit_width(std::max<std::size_t>(n, 2) - 1)));
}

// Scratch words for inverse() of n coefficients.
inline std::size_t inverse_scratch(std::size_t n) { return 2 * Arena::footprint(std::size_t(1) << inverse_log(n)); }

// g = 1 / f mod x^n for n = g.size() >= 1. f[0] != 0; coefficients of f past f.size() are zero.
// scratch: inverse_scratch(n) words, 32-byte aligned (from an Arena). t: lg_max >= inverse_log(n).
//
// Newton steps double the known prefix g_k: with transforms of length 2k,
//   e = f g_k mod (x^2k - 1), whose coefficients [k, 2k) are those of f g_k;
//   g[k, 2k) = -(x^k e[k, 2k) g_k mod (x^2k - 1))[k, 2k).
// Each step takes 5 transforms of length 2k (g_k's is used twice).
inline void inverse(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                    std::span<std::uint32_t> scratch) {
    const std::size_t n = g.size();
    std::size_t k = std::min(n, detail::kInverseBase);
    detail::inverse_direct(f, g.first(k));
    for (; k < n; k *= 2) {
        const std::size_t len = 2 * k, end = std::min(len, n);
        const std::span<std::uint32_t> gk = scratch.first(len), e = scratch.subspan(Arena::footprint(len), len);
        t.forward(g.first(k), 0, gk);
        t.cyclic_product(f.first(std::min(len, f.size())), 0, e, gk, Half::kUpper);
        t.cyclic_product(e.subspan(k), k, e, gk, Half::kUpper);
        for (std::size_t i = k; i < end; ++i) g[i] = e[i] ? detail::kP - e[i] : 0;
    }
}

}  // namespace poly
