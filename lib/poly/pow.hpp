// Powers of a power series modulo 998244353: g = c (f / f[0])^e mod x^n, that is
// c exp(e log(f / f[0])), for any residue e. Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::power_log(n)) + poly::power_scratch(n) + ...);
//   poly::Transform t(arena, poly::power_log(n));
//   poly::power(t, f, e, c, g, arena.take(poly::power_scratch(n)));  // g.size() == n; g may be f
//
// For an integer M >= 0 and n <= P, f^M = power(f, M mod P, f[0]^M): (f / f[0])^M has constant
// term 1, so it depends on M mod P only. A square root of f is power(f, 1 / 2, sqrt(f[0])).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/exp.hpp"
#include "lib/poly/log.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// out[i] = a[i] c for i < out.size() = a.size(), any alignment; out may be a.
inline void scale(std::span<const std::uint32_t> a, std::uint32_t c, std::span<std::uint32_t> out) {
    const Factor factor(c);
    const std::size_t full = a.size() / 8 * 8;
    for (std::size_t i = 0; i < full; i += 8) store_unaligned(out.data() + i, reduce(times(load_unaligned(a.data() + i), factor), kP));
    for (std::size_t i = full; i < a.size(); ++i) out[i] = ntt::detail::multiply_mod(a[i], c);
}

}  // namespace detail

// Transform length power uses for n coefficients: the Transform needs lg_max >= this.
inline int power_log(std::size_t n) { return std::max(log_log(n), exp_log(n)); }

// Scratch words for power() of n coefficients.
inline std::size_t power_scratch(std::size_t n) { return std::max(log_scratch(n), exp_scratch(n)); }

// g = c exp(e log(f / f[0])) mod x^n for n = g.size() >= 1, f[0] != 0, e and c residues.
// Coefficients of f past f.size() are zero. g may be f; otherwise the two must not overlap.
// scratch: power_scratch(n) words, 32-byte aligned (from an Arena). t: lg_max >= power_log(n).
inline void power(const Transform& t, std::span<const std::uint32_t> f, std::uint32_t e, std::uint32_t c,
                  std::span<std::uint32_t> g, std::span<std::uint32_t> scratch) {
    const std::size_t n = g.size(), size = std::min(f.size(), n);
    if (f.empty() || f[0] == 0) std::abort();
    detail::scale(f.first(size), detail::scalar_inverse(f[0]), g.first(size));
    std::fill(g.begin() + size, g.end(), 0);
    log(t, g, g, scratch);
    detail::scale(g, e, g);
    exp(t, g, g, scratch);
    detail::scale(g, c, g);
}

}  // namespace poly
