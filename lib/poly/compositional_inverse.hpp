// Compositional inverse of a power series modulo 998244353: g with f(g) = g(f) = x mod x^n, for
// f[0] = 0 and f[1] != 0, by Lagrange inversion from power projection. x86-64 with AVX2.
// Design: lib/poly/notes.md (Compositional inverse).
//
//   const std::size_t n = ...;
//   const int lg = poly::compositional_inverse_log(n);
//   poly::Arena arena(poly::Transform::words(lg) + poly::compositional_inverse_scratch(n) + ...);
//   poly::Transform t(arena, lg);
//   poly::compositional_inverse(t, f, g, arena.take(poly::compositional_inverse_scratch(n)));  // g.size() == n
//
// Lagrange: (n - 1) [x^(n-1)] f^k = k [x^(n-1-k)] (x / g)^(n-1) for 0 < k < n. power_projection
// gives a[k] = [x^(n-1)] f^k, so H = sum_(j < n-1) a[n-1-j] / (n-1-j) x^j = (x / g)^(n-1) / (n - 1)
// mod x^(n-1), and g / x = (H / H[0])^(-1 / (n-1)) / f[1] mod x^(n-1) (power, which divides by H[0]).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/pow.hpp"
#include "lib/poly/projection.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

// Transform length compositional_inverse() uses for n coefficients: the Transform needs lg_max >= this.
inline int compositional_inverse_log(std::size_t n) {
    return std::max(projection_log(n), power_log(std::max<std::size_t>(n, 2) - 1));
}

// Scratch words for compositional_inverse() of n coefficients.
inline std::size_t compositional_inverse_scratch(std::size_t n) {
    const std::size_t rest = std::max<std::size_t>(n, 2) - 1;
    return Arena::footprint(n) + std::max(projection_scratch(n), Arena::footprint(rest) + power_scratch(rest));
}

// g = the compositional inverse of f mod x^n for n = g.size() >= 1, f[0] = 0 and f[1] != 0;
// coefficients past f.size() are zero. scratch: compositional_inverse_scratch(n) words, 32-byte
// aligned (from an Arena). t: lg_max >= compositional_inverse_log(n). f and g must not overlap.
inline void compositional_inverse(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                                  std::span<std::uint32_t> scratch) {
    const auto reciprocal = [](std::uint32_t x) { return ntt::detail::power(x, kModulus - 2); };
    const std::size_t n = g.size();
    if (f.size() < 2 || f[0] != 0 || f[1] == 0) std::abort();
    g[0] = 0;
    if (n == 1) return;
    const std::span<std::uint32_t> a = scratch.first(n), rest = scratch.subspan(Arena::footprint(n));
    const std::span<std::uint32_t> h = rest.first(n - 1);  // after the projection
    power_projection(t, f, a, rest.first(projection_scratch(n)));
    divide_by_index(a.subspan(1), 1, h);  // a[i + 1] / (i + 1) = H[n - 2 - i]
    std::reverse(h.begin(), h.end());
    const std::uint32_t e = kModulus - reciprocal(std::uint32_t(n - 1)), c = reciprocal(f[1]);
    power(t, h, e, c, g.subspan(1), rest.subspan(Arena::footprint(n - 1), power_scratch(n - 1)));
}

}  // namespace poly
