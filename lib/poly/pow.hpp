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

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/exp.hpp"
#include "lib/poly/log.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// out = e x (a + s b) + (alpha + s beta) for transforms a, b of length n = out.size() >= 128
// (b only if kSum) and alpha, beta <= P, where s = x^(n/2) is 1 on the lower half of the leaves
// and -1 on the upper half. Leaf p of x a is (w_p a_7, a_0, .., a_6): per 8 leaves, the lanes 7
// are gathered into one vector for their products with w_p. out may be a or b.
template <bool kSum>
void times_x(const std::uint32_t* roots, const std::uint32_t* a, const std::uint32_t* b, std::uint32_t e,
             std::uint32_t alpha, std::uint32_t beta, std::span<std::uint32_t> out) {
    const std::size_t leaves = out.size() / 8;
    const Factor scale(e);
    const Vec rotate = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6), odd = _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1);
    const std::uint32_t constants[2] = {(alpha + beta) % kP, (alpha + kP - beta) % kP};
    for (std::size_t p = 0; p < leaves; p += 8) {
        const bool upper = p >= leaves / 2;
        // w_p for leaves p .. p + 7: r[p/2 + i] for even p, -r[p/2 + i] (quotient ~q) for odd p
        const Factors r = entries(roots, p / 16 * 8);
        const Vec pick = p % 16 ? _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7) : _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
        const Vec rw = _mm256_permutevar8x32_epi32(r.w, pick), rq = _mm256_permutevar8x32_epi32(r.q, pick);
        const Factors w{_mm256_blend_epi32(rw, _mm256_sub_epi32(broadcast(kP), rw), 0xAA), _mm256_xor_si256(rq, odd)};
        Vec y[8];  // e (a + s b), canonical
#pragma GCC unroll 8
        for (std::size_t i = 0; i < 8; ++i) {
            Vec c = load(a + 8 * (p + i));
            if constexpr (kSum) c = upper ? difference(c, load(b + 8 * (p + i))) : reduce(add(c, load(b + 8 * (p + i))), kP);
            y[i] = reduce(times(c, scale), kP);
        }
        // lane i of z: lane 7 of y[i] times w_(p+i), plus the constant
        const Vec t01 = _mm256_unpackhi_epi32(y[0], y[1]), t23 = _mm256_unpackhi_epi32(y[2], y[3]);
        const Vec t45 = _mm256_unpackhi_epi32(y[4], y[5]), t67 = _mm256_unpackhi_epi32(y[6], y[7]);
        const Vec u03 = _mm256_unpackhi_epi64(t01, t23), u47 = _mm256_unpackhi_epi64(t45, t67);
        const Vec z = canonical(add(times(_mm256_permute2x128_si256(u03, u47, 0x31), w), broadcast(constants[upper])));
        const Vec zh = _mm256_permute2x128_si256(z, z, 0x11);
        const Vec lane0[8] = {z,  _mm256_srli_si256(z, 4),  _mm256_srli_si256(z, 8),  _mm256_srli_si256(z, 12),
                              zh, _mm256_srli_si256(zh, 4), _mm256_srli_si256(zh, 8), _mm256_srli_si256(zh, 12)};
#pragma GCC unroll 8
        for (std::size_t i = 0; i < 8; ++i)
            store(out.data() + 8 * (p + i), _mm256_blend_epi32(_mm256_permutevar8x32_epi32(y[i], rotate), lane0[i], 1));
    }
}

}  // namespace detail

// Transform length power uses for n coefficients: the Transform needs lg_max >= this.
inline int power_log(std::size_t n) { return std::max(detail::log_derivative_log(n), exp_log(n)); }

// Scratch words for power() of n coefficients.
inline std::size_t power_scratch(std::size_t n) {
    return 3 * Arena::footprint(detail::exp_length(n) / 2) +
           std::max(detail::log_derivative_scratch(n), detail::exp_newton_scratch(n));
}

// g = c exp(e log(f / f[0])) mod x^n for n = g.size() >= 1, f[0] != 0, e and c residues.
// Coefficients of f past f.size() are zero. g may be f; otherwise the two must not overlap.
// scratch: power_scratch(n) words, 32-byte aligned (from an Arena). t: lg_max >= power_log(n).
//
// g solves x g' = q g with q = e x f'/f and g[0] = c: f'/f by the blocked division of log, then
// the Newton steps of exp, which do not depend on the scale of g. q is kept as exp_newton takes
// it: below m = exp_length(n) / 2 in the scratch, from m on in g, where log_derivative no longer
// reads f. When the division's blocks d_j of f'/f have k = m/2 coefficients, the last step's
// transforms of length m of q mod x^m and of q[m, m + k) come from the division's T(d_0), T(d_1)
// and T(d_2): q mod x^m = e x (d_0 + x^k d_1) - q[m] x^m and q[m, m + k) = q[m] + e x d_2 -
// q[m + k] x^k, with x^m = 1 on every leaf.
inline void power(const Transform& t, std::span<const std::uint32_t> f, std::uint32_t e, std::uint32_t c,
                  std::span<std::uint32_t> g, std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size(), m = exp_length(n) / 2, k = m / 2, stride = Arena::footprint(m);
    if (f.empty() || f[0] == 0) std::abort();
    const std::span<std::uint32_t> lo = scratch.first(m), rest = scratch.subspan(3 * stride);
    const auto q = [&](std::size_t i) { return i < m ? lo.data() + i : g.data() + i; };
    const Factor factor(e);
    // T_m(q mod x^m) and T_m(q[m, m + k)), if the division has the transforms they come from
    const bool reuse = m >= 128 && log_block(n) == k && log_blocks(n) >= 3;
    const bool block2 = reuse && log_blocks(n) == 4 && n - m > k;
    const std::span<std::uint32_t> qt = scratch.subspan(stride, reuse ? m : 0);
    const std::span<std::uint32_t> q2t = scratch.subspan(2 * stride, block2 ? m : 0);
    const std::uint32_t* d0t = nullptr;
    lo[0] = 0;
    if (n >= 2)
        log_derivative(
            t, f, n, rest,
            [&](std::size_t first, std::span<const std::uint32_t> d) {  // q[first + 1 + i] = e d[i]
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
            },
            [&](std::size_t j, std::span<const std::uint32_t> dt) {  // dt = T(d_j)
                if (j == 0) d0t = dt.data();
                if (j == 1 && reuse) times_x<true>(t.roots(), d0t, dt.data(), e, kP - *q(m), 0, qt);
                if (j == 2 && block2) times_x<false>(t.roots(), dt.data(), nullptr, e, *q(m), kP - *q(m + k), q2t);
            });
    const std::span<std::uint32_t> start = g.first(std::min(n, kExpBase));
    exp_direct(lo, start);
    for (std::uint32_t& x : start) x = ntt::detail::multiply_mod(x, c);
    if (n > kExpBase) exp_newton(t, lo, g, rest, QTransforms{qt, q2t});
}

}  // namespace poly
