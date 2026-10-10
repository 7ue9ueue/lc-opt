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
    return Arena::footprint(detail::exp_length(n) / 2) + std::max(detail::log_derivative_scratch(n), detail::exp_newton_scratch(n));
}

// g = c exp(e log(f / f[0])) mod x^n for n = g.size() >= 1, f[0] != 0, e and c residues.
// Coefficients of f past f.size() are zero. g may be f; otherwise the two must not overlap.
// scratch: power_scratch(n) words, 32-byte aligned (from an Arena). t: lg_max >= power_log(n).
//
// g solves x g' = q g with q = e x f'/f and g[0] = c: f'/f by the blocked division of log, then
// the Newton steps of exp, which do not depend on the scale of g. q is kept as exp_newton takes
// it: below m = exp_length(n) / 2 in the scratch, from m on in g, where log_derivative no longer
// reads f.
inline void power(const Transform& t, std::span<const std::uint32_t> f, std::uint32_t e, std::uint32_t c,
                  std::span<std::uint32_t> g, std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size(), m = exp_length(n) / 2;
    if (f.empty() || f[0] == 0) std::abort();
    const std::span<std::uint32_t> lo = scratch.first(m), rest = scratch.subspan(Arena::footprint(m));
    const auto q = [&](std::size_t i) { return i < m ? lo.data() + i : g.data() + i; };
    const Factor factor(e);
    lo[0] = 0;
    if (n >= 2)
        log_derivative(t, f, n, rest, [&](std::size_t first, std::span<const std::uint32_t> d) {  // q[first + 1 + i] = e d[i]
            std::size_t i = 0;
            for (; i + 8 <= d.size(); i += 8) {
                const std::size_t at = first + 1 + i;
                const Vec x = reduce(times(load(d.data() + i), factor), kP);
                if (at + 8 <= m || at >= m) {
                    store_unaligned(q(at), x);
                } else {
                    alignas(32) std::uint32_t words[8];
                    store(words, x);
                    for (std::size_t j = 0; j < 8; ++j) *q(at + j) = words[j];
                }
            }
            for (; i < d.size(); ++i) *q(first + 1 + i) = ntt::detail::multiply_mod(d[i], e);
        });
    const std::span<std::uint32_t> start = g.first(std::min(n, kExpBase));
    exp_direct(lo, start);
    for (std::uint32_t& x : start) x = ntt::detail::multiply_mod(x, c);
    if (n > kExpBase) exp_newton(t, lo, g, rest);
}

}  // namespace poly
