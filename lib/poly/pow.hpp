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

#include "lib/poly/exp.hpp"
#include "lib/poly/log.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

// Transform length power uses for n coefficients: the Transform needs lg_max >= this.
inline int power_log(std::size_t n) { return std::max(detail::log_derivative_log(n), exp_log(n)); }

// Scratch words for power() of n coefficients.
inline std::size_t power_scratch(std::size_t n) {
    return Arena::footprint(detail::exp_length(n)) + std::max(detail::log_derivative_scratch(n), detail::exp_newton_scratch(n));
}

// g = c exp(e log(f / f[0])) mod x^n for n = g.size() >= 1, f[0] != 0, e and c residues.
// Coefficients of f past f.size() are zero. g may be f; otherwise the two must not overlap.
// scratch: power_scratch(n) words, 32-byte aligned (from an Arena). t: lg_max >= power_log(n).
//
// g solves g' = d g with d = e f'/f and g[0] = c: d by the blocked division of log, then the
// Newton steps of exp, which do not depend on the scale of g.
inline void power(const Transform& t, std::span<const std::uint32_t> f, std::uint32_t e, std::uint32_t c,
                  std::span<std::uint32_t> g, std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size(), len = exp_length(n);
    if (f.empty() || f[0] == 0) std::abort();
    const std::span<std::uint32_t> d = scratch.first(len), rest = scratch.subspan(Arena::footprint(len));
    const Factor factor(e);
    if (n >= 2)
        log_derivative(t, f, n, rest, [&](std::size_t first, std::span<const std::uint32_t> q) {
            for (std::size_t i = 0; i < q.size(); i += 8) store(d.data() + first + i, reduce(times(load(q.data() + i), factor), kP));
        });
    std::fill(d.begin() + std::ptrdiff_t(n - 1), d.end(), 0);
    const std::span<std::uint32_t> start = g.first(std::min(n, kExpBase));
    exp_direct(d, start);
    for (std::uint32_t& x : start) x = ntt::detail::multiply_mod(x, c);
    if (n > kExpBase) exp_newton(t, d, g, rest);
}

}  // namespace poly
