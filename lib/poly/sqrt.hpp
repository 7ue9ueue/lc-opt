// Square root of a power series modulo 998244353: g = sqrt(f) mod x^n with g[0] = c, where
// c^2 = f[0] != 0. Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::sqrt_log(n)) + poly::sqrt_scratch(n) + ...);
//   poly::Transform t(arena, poly::sqrt_log(n));
//   poly::sqrt(t, f, c, g, arena.take(poly::sqrt_scratch(n)));  // g.size() == n
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// Coefficients computed directly before the Newton steps; the first step uses transforms of
// length kSqrtBase.
inline constexpr std::size_t kSqrtBase = 64;

// g = sqrt(f) mod x^n with g[0] = c, n = g.size() <= kSqrtBase: 2 c g_i = f_i - sum_(0<j<i) g_j g_(i-j).
inline void sqrt_direct(std::span<const std::uint32_t> f, std::uint32_t c, std::span<std::uint32_t> g) {
    using ntt::detail::multiply_mod;
    const std::uint32_t inverse = ntt::detail::power(multiply_mod(2, c), kP - 2);  // 1 / (2 c)
    g[0] = c;
    for (std::size_t i = 1; i < g.size(); ++i) {
        std::uint64_t sum = i < f.size() ? f[i] : 0;  // fewer than kSqrtBase terms <= P
        for (std::size_t j = 1; j < i; ++j) sum += kP - std::uint64_t(g[j]) * g[i - j] % kP;
        g[i] = multiply_mod(std::uint32_t(sum % kP), inverse);
    }
}

// f[at, at + 8), zero past f.size().
inline Vec load_padded(std::span<const std::uint32_t> f, std::size_t at) {
    if (at + 8 <= f.size()) return load_unaligned(f.data() + at);
    alignas(32) std::uint32_t x[8] = {};
    if (at < f.size()) std::copy(f.begin() + std::ptrdiff_t(at), f.end(), x);
    return load(x);
}

// out[i] = (s[i] - f[first + i] - f[m + first + i]) / 2 for i < count, rounded up to a multiple of
// 8. For s = g^2 mod (x^m - 1) and g^2 = f mod x^m, s - f[0, m) = (g^2)[m, 2m), so out holds
// (g^2 - f)[m + first, m + first + count) / 2. s canonical; s and out 32-byte aligned; out may be s.
inline void half_residual(std::span<const std::uint32_t> f, std::size_t m, std::size_t first, const std::uint32_t* s,
                          std::uint32_t* out, std::size_t count) {
    const Factor half((kP + 1) / 2);
    for (std::size_t i = 0; i < count; i += 8) {
        const Vec fs = add(load_padded(f, first + i), load_padded(f, m + first + i));
        const Vec x = _mm256_sub_epi32(add(load(s + i), broadcast(2 * kP)), fs);  // < 3P
        store(out + i, reduce(times(x, half), kP));
    }
}

}  // namespace detail

// Transform length sqrt uses for n coefficients: the Transform needs lg_max >= this.
inline int sqrt_log(std::size_t n) {
    return std::max(Transform::kMinLog, int(std::bit_width(std::max<std::size_t>(n, 2) - 1)) - 1);
}

// Scratch words for sqrt() of n coefficients.
inline std::size_t sqrt_scratch(std::size_t n) {
    if (n <= detail::kSqrtBase) return 0;
    const std::size_t len = std::size_t(1) << sqrt_log(n);
    return 3 * Arena::footprint(len) + Arena::footprint(len / 2);
}

// g = sqrt(f) mod x^n for n = g.size() >= 1 with g[0] = c; c^2 = f[0] != 0. Coefficients of f
// past f.size() are zero. f and g must not overlap. scratch: sqrt_scratch(n) words, 32-byte
// aligned (from an Arena). t: lg_max >= sqrt_log(n).
//
// Newton steps double the known prefix g = sqrt(f) mod x^m, keeping h = -1 / g mod x^(m/2) and
// its transform of length m. With transforms of length m and 2m:
//   h = h + x^(m/2) (h (g h)[m/2, m) mod x^(m/2))   now h = -1 / g mod x^m
//   r = (g^2 - f)[m, 2m) / 2                         from g^2 mod (x^m - 1) - f mod x^m
//   g[m, 2m) = h r mod x^m
// The last step, from m to n <= 2m, keeps h at x^(m/2) and uses transforms of length m only:
// with r0 = r mod x^(m/2), d0 = h r0 mod x^(m/2), and d1 = h (r + g d0)[m/2, n - m) mod x^(n - 3m/2),
// g[m, n) = d0 + x^(m/2) d1 (Karp and Markstein's division).
// Per step: 11 transforms of length m; the last step 8.
inline void sqrt(const Transform& t, std::span<const std::uint32_t> f, std::uint32_t c, std::span<std::uint32_t> g,
                 std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size();
    if (f.empty() || f[0] == 0 || ntt::detail::multiply_mod(c, c) != f[0]) std::abort();
    sqrt_direct(f, c, g.first(std::min(n, kSqrtBase)));
    if (n <= kSqrtBase) return;
    const std::size_t len = std::size_t(1) << sqrt_log(n);
    const auto take = [&scratch](std::size_t words) {
        const std::span<std::uint32_t> s = scratch.first(words);
        scratch = scratch.subspan(Arena::footprint(words));
        return s;
    };
    const std::span<std::uint32_t> gt = take(len), ht = take(len), w = take(len), h = take(len / 2);
    std::size_t m = kSqrtBase;
    inverse_direct(g.first(m / 2), h.first(m / 2));
    for (std::uint32_t& x : h.first(m / 2)) x = x ? kP - x : 0;
    t.forward(h.first(m / 2), 0, ht.first(m));
    for (; 2 * m < n; m *= 2) {
        const std::size_t half = m / 2;
        const std::span<std::uint32_t> g_low = gt.first(m), h_low = ht.first(m), w_low = w.first(m);
        t.forward(g.first(m), 0, g_low);
        // h[m/2, m) = h e mod x^(m/2), e = (g h)[m/2, m)
        t.inverse_product(g_low, h_low, w_low, Half::kUpper);
        t.cyclic_product(w_low.subspan(half), half, w_low, h_low, Half::kUpper);
        std::copy_n(w.begin() + std::ptrdiff_t(half), half, h.begin() + std::ptrdiff_t(half));
        // r at w[0, m), then g[m, 2m) = h r mod x^m by transforms of length 2m
        t.inverse_product(g_low, g_low, w_low);
        half_residual(f, m, 0, w.data(), w.data(), m);
        t.forward(h.first(m), 0, ht.first(2 * m));
        t.cyclic_product(w_low, 0, w.first(2 * m), ht.first(2 * m), Half::kLower);
        std::copy_n(w.begin(), m, g.begin() + std::ptrdiff_t(m));
    }
    const std::size_t half = m / 2, rest = n - m;
    const std::span<std::uint32_t> g_low = gt.first(m), h_low = ht.first(m), w_low = w.first(m);
    t.forward(g.first(m), 0, g_low);
    t.inverse_product(g_low, g_low, w_low, rest <= half ? Half::kLower : Half::kBoth);
    if (rest <= half) {
        half_residual(f, m, 0, w.data(), w.data(), rest);
        t.cyclic_product(w.first(rest), 0, w_low, h_low, Half::kLower);
        std::copy_n(w.begin(), rest, g.begin() + std::ptrdiff_t(m));
        return;
    }
    // r0 at w[0, m/2), r[m/2, n - m) at h (no longer needed: h_low holds its transform)
    half_residual(f, m, 0, w.data(), w.data(), half);
    half_residual(f, m, half, w.data() + half, h.data(), rest - half);
    t.cyclic_product(w.first(half), 0, w_low, h_low, Half::kLower);
    std::copy_n(w.begin(), half, g.begin() + std::ptrdiff_t(m));
    t.cyclic_product(w.first(half), 0, w_low, g_low, Half::kUpper);  // (g d0)[m/2, m)
    for (std::size_t i = 0; i < rest - half; i += 8)
        store(h.data() + i, reduce(add(load(h.data() + i), load(w.data() + half + i)), kP));
    t.cyclic_product(h.first(rest - half), 0, w_low, h_low, Half::kLower);
    std::copy_n(w.begin(), rest - half, g.begin() + std::ptrdiff_t(m + half));
}

}  // namespace poly
