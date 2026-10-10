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

// One Newton step: to = (1 / f)[k, k + to.size()) from g = 1 / f mod x^k, to.size() <= k, with
// transforms of length 2k (t: lg_max >= log2(2k)):
//   e = f g mod (x^2k - 1), whose coefficients [k, 2k) are those of f g;
//   to = -(x^k e[k, 2k) g mod (x^2k - 1))[k, k + to.size()).
// 5 transforms of length 2k (g's is used twice). gt, work: 2k words each, 32-byte aligned. f may
// start at work (f is then overwritten); to may start at work[k] (no copy is made then). Otherwise
// none may overlap. (Writing the transform's output half straight into to was slower on lc-intel
// than this copy: lib/poly/notes.md.)
inline void inverse_step(const Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                         std::span<std::uint32_t> to, std::span<std::uint32_t> gt, std::span<std::uint32_t> work) {
    const std::size_t k = g.size();
    t.forward(g, 0, gt);
    t.cyclic_product(f.first(std::min(2 * k, f.size())), 0, work, gt, Half::kUpper);
    t.cyclic_product(work.subspan(k), k, work, gt, Half::kUpper, detail::kP - 1);
    if (to.data() != work.data() + k) std::copy_n(work.begin() + std::ptrdiff_t(k), to.size(), to.begin());
}

// g = 1 / f mod x^n for n = g.size() >= 1. f[0] != 0; coefficients of f past f.size() are zero.
// scratch: inverse_scratch(n) words, 32-byte aligned (from an Arena). t: lg_max >= inverse_log(n).
// Newton steps from kInverseBase, each doubling the known prefix of g.
inline void inverse(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                    std::span<std::uint32_t> scratch) {
    const std::size_t n = g.size();
    std::size_t k = std::min(n, detail::kInverseBase);
    detail::inverse_direct(f, g.first(k));
    for (; k < n; k *= 2) {
        const std::size_t len = 2 * k;
        inverse_step(t, f, g.first(k), g.subspan(k, std::min(k, n - k)), scratch.first(len),
                     scratch.subspan(Arena::footprint(len), len));
    }
}

}  // namespace poly
