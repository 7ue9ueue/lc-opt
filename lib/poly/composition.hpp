// Composition of power series modulo 998244353: h = f(g) mod x^n for g[0] = 0, by Kinoshita and
// Li's algorithm (the transpose of power projection). Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::compose_log(n)) + poly::compose_scratch(n) + ...);
//   poly::Transform t(arena, poly::compose_log(n));
//   poly::compose(t, f, g, h, arena.take(poly::compose_scratch(n)));  // h.size() == n
//
// The levels (shared with power projection): m = 2^T >= n, Q_0(x, y) = 1 - y g(x), and
// Q_(s+1)(x^2, y) = Q_s(x, y) Q_s(-x, y) mod x^(m / 2^s). Q_s has m / 2^s coefficients in x and degree
// 2^s in y; Q_s(x, 0) = Q_s(0, y) = 1. A bivariate polynomial is stored by Kronecker substitution
// x = z, y = z^stride, so that a product mod (z^len - 1) wraps y mod y^(len / stride) and
// carries nothing from x into y while the x degree stays below stride.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// Up to this many coefficients, Horner's rule.
inline constexpr std::size_t kComposeBase = 32;

// h = f(g) mod x^n by Horner's rule, n = h.size() <= kComposeBase. h = h g + f_i from the top
// coefficient down, so h[k] reads only h[0, k) (g[0] = 0).
inline void compose_direct(std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                           std::span<std::uint32_t> h) {
    const std::size_t n = h.size();
    std::fill(h.begin(), h.end(), 0);
    for (std::size_t i = std::min(f.size(), n); i-- > 0;) {
        for (std::size_t k = n; k-- > 1;) {
            std::uint64_t sum = 0;  // fewer than kComposeBase terms < P
            for (std::size_t j = 1; j <= k && j < g.size(); ++j) sum += std::uint64_t(g[j]) * h[k - j] % kP;
            h[k] = std::uint32_t(sum % kP);
        }
        h[0] = f[i];
    }
}

// x / 2 mod P in [0, P) for x < 2P.
inline Vec halve(Vec x) {
    const Vec odd = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_and_si256(x, broadcast(1)));
    return reduce(_mm256_add_epi32(_mm256_srli_epi32(x, 1), _mm256_and_si256(odd, broadcast((kP + 1) / 2))), kP);
}

// -x mod P in [0, P) for x < P.
inline Vec negate(Vec x) {
    return _mm256_min_epu32(_mm256_sub_epi32(broadcast(kP), x), _mm256_sub_epi32(_mm256_setzero_si256(), x));
}

// The parts of Q(z) = E(z^2) + z O(z^2) from its transform qh of length len: the transforms of
// length len / 2 of E, -O and u O (u = z^2). Leaves 2p and 2p + 1 of qh are Q mod (z^8 - s) and
// Q mod (z^8 + s) with s = r[p]; together Q mod (z^16 - s^2) = lo + z^8 hi, lo = (a + b) / 2,
// hi = (a - b) / (2 s). Since s^2 = w_p, the even and odd coefficients of lo + z^8 hi are leaf p
// of E and O. Tables: roots with len / 32 entries, inverse_roots with len / 16.
inline void split_parts(const std::uint32_t* qh, std::size_t len, const std::uint32_t* roots,
                        const std::uint32_t* inverse_roots, std::uint32_t* even, std::uint32_t* odd_negated,
                        std::uint32_t* odd_shifted) {
    const Vec deal = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7), rotate = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    for (std::size_t p = 0; p < len / 16; ++p) {
        const Vec a = load(qh + 16 * p), b = load(qh + 16 * p + 8);
        const Vec lo = _mm256_permutevar8x32_epi32(halve(add(a, b)), deal);
        const Vec hi = _mm256_permutevar8x32_epi32(halve(times(diff(a, b), entry(inverse_roots, p))), deal);
        const Vec e = _mm256_permute2x128_si256(lo, hi, 0x20), o = _mm256_permute2x128_si256(lo, hi, 0x31);
        store(even + 8 * p, e);
        store(odd_negated + 8 * p, negate(o));
        const Vec r = _mm256_permutevar8x32_epi32(o, rotate);  // u o mod (u^8 - w_p): lane 0 is w_p o_7
        store(odd_shifted + 8 * p, _mm256_blend_epi32(r, reduce(times(r, leaf_weight(roots, p)), kP), 1));
    }
}

// Bump allocation of 32-byte aligned spans from scratch, with Arena's gaps.
class Carve {
public:
    explicit Carve(std::span<std::uint32_t> scratch) : rest_(scratch) {}

    std::span<std::uint32_t> take(std::size_t n) {
        const std::size_t step = Arena::footprint(n);
        if (step > rest_.size()) std::abort();
        const std::span<std::uint32_t> s = rest_.first(n);
        rest_ = rest_.subspan(step);
        return s;
    }

private:
    std::span<std::uint32_t> rest_;
};

// The levels of g: for s < T, the transforms of length 2m of E_s and -O_s, Q_s = E_s(x^2, y) +
// x O_s(x^2, y), with Kronecker stride m / 2^s (2^(s+1) rows in y).
struct Levels {
    std::size_t m;  // 2^T >= 32
    int lg;         // log2(m)
    std::span<std::uint32_t> roots, inverse_roots;  // leaf weights for transforms of length 4m
    std::span<std::uint32_t> parts;                 // level s: E_s at [4ms, 4ms + 2m), -O_s after it
    std::span<std::uint32_t> work[4];               // lengths 4m, 2m, 2m, 2m

    std::span<std::uint32_t> even(int s) const { return parts.subspan(2 * std::size_t(s) * part(m), 2 * m); }
    std::span<std::uint32_t> odd(int s) const { return parts.subspan((2 * std::size_t(s) + 1) * part(m), 2 * m); }

    static std::size_t part(std::size_t m) { return Arena::footprint(2 * m); }

    static std::size_t scratch(std::size_t m) {
        const int lg = std::countr_zero(m);
        return Arena::footprint(ntt::detail::table_words(lg + 1)) + Arena::footprint(ntt::detail::table_words(lg + 2)) +
               Arena::footprint(2 * std::size_t(lg) * part(m)) + Arena::footprint(4 * m) + 3 * part(m);
    }

    Levels(std::size_t size, std::span<std::uint32_t> scratch) : m(size), lg(std::countr_zero(size)) {
        Carve carve(scratch);
        roots = carve.take(ntt::detail::table_words(lg + 1));
        inverse_roots = carve.take(ntt::detail::table_words(lg + 2));
        ntt::detail::build_table(roots.data(), m / 8, ntt::detail::kRoots[0]);
        ntt::detail::build_table(inverse_roots.data(), m / 4, ntt::detail::kRoots[1]);
        parts = carve.take(2 * std::size_t(lg) * part(m));
        work[0] = carve.take(4 * m);
        for (int i = 1; i < 4; ++i) work[i] = carve.take(2 * m);
    }
};

// Q_0 = 1 - y g(x) at stride 2m: a[0] = 1, a[2m + j] = -g[j].
inline void first_level(std::span<const std::uint32_t> g, std::size_t m, std::span<std::uint32_t> a) {
    std::fill(a.begin(), a.end(), 0);
    a[0] = 1;
    for (std::size_t j = 1; j < std::min(g.size(), m); ++j) a[2 * m + j] = g[j] ? kP - g[j] : 0;
}

// From v = Q_s(x) Q_s(-x) mod (z^(2m) - 1) at stride L (rows 2Y wrapped: row 0 holds 1 + the row
// 2Y), Q_(s+1) at stride L (4Y rows of which 0 .. 2Y are used, x below L / 2) in a, length 4m.
inline void next_level(std::span<std::uint32_t> a, std::size_t stride, std::size_t rows) {
    const std::size_t half = stride / 2, top = rows * stride;
    std::copy(a.begin(), a.begin() + std::ptrdiff_t(half), a.begin() + std::ptrdiff_t(top));
    a[top] = a[top] ? a[top] - 1 : kP - 1;
    std::fill(a.begin() + std::ptrdiff_t(top + half), a.end(), 0);
    a[0] = 1;
    std::fill(a.begin() + 1, a.begin() + std::ptrdiff_t(half), 0);
    for (std::size_t i = 0; i < rows; ++i)
        std::fill(a.begin() + std::ptrdiff_t(i * stride + half), a.begin() + std::ptrdiff_t((i + 1) * stride), 0);
}

// The transforms of all levels of g into levels.parts. t: lg_max >= levels.lg + 2.
inline void build_levels(const Transform& t, std::span<const std::uint32_t> g, Levels& levels) {
    const std::size_t m = levels.m;
    const std::span<std::uint32_t> a = levels.work[0], shifted = levels.work[1];
    first_level(g, m, a);
    for (int s = 0; s < levels.lg; ++s) {
        t.forward(a);
        split_parts(a.data(), 4 * m, levels.roots.data(), levels.inverse_roots.data(), levels.even(s).data(),
                    levels.odd(s).data(), shifted.data());
        if (s + 1 == levels.lg) break;
        // Q_(s+1) = E_s^2 - u O_s^2 = E_s E_s + (u O_s)(-O_s) at stride m / 2^s, 2^(s+1) rows (wrapped).
        const Transform::Pair pairs[2] = {{levels.even(s), levels.even(s)}, {shifted, levels.odd(s)}};
        t.inverse_product_sum(pairs, a.first(2 * m));
        next_level(a, m >> s, std::size_t(2) << s);
    }
}

}  // namespace detail

// Transform length compose() uses for n coefficients: the Transform needs lg_max >= this.
inline int compose_log(std::size_t n) {
    if (n <= detail::kComposeBase) return Transform::kMinLog;
    return int(std::bit_width(n - 1)) + 2;
}

// Scratch words for compose() of n coefficients.
inline std::size_t compose_scratch(std::size_t n) {
    if (n <= detail::kComposeBase) return 0;
    return detail::Levels::scratch(std::bit_ceil(n));
}

// h = f(g) mod x^n for n = h.size() >= 1 and g[0] = 0 (if g is not empty); coefficients past
// f.size() and g.size() are zero. scratch: compose_scratch(n) words, 32-byte aligned (from an
// Arena). t: lg_max >= compose_log(n). f, g and h must not overlap.
//
// Transposed power projection: with m = 2^T >= n, h_rev = the transpose of w -> ([x^(m-1)] w g^i)_i
// applied to f. Level s maps P_(s+1) (m / 2^(s+1) by 2^(s+1) coefficients) to P_s (m / 2^s by
// 2^s): P_s[a][b] = sum P_(s+1)[(a + c) / 2][b + e] Q_s(-x)[c][e] over a + c odd. Split by the parity
// of a, P_s[2a + 1] and P_s[2a] are middle products of P_(s+1) with E_s and -O_s; with P_(s+1)
// reversed in x and y, they are cyclic products of length 2m read at reversed indices, and the
// reversals cancel between levels. P_T = f (one row in x), h[k] = P_0[m - 1 - k][0].
inline void compose(const Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                    std::span<std::uint32_t> h, std::span<std::uint32_t> scratch) {
    const std::size_t n = h.size();
    if (n <= detail::kComposeBase) return detail::compose_direct(f, g, h);
    detail::Levels levels(std::bit_ceil(n), scratch);
    detail::build_levels(t, g, levels);
    const std::size_t m = levels.m;
    // Level T - 1 reads f reversed at stride 2: x below 1, 2^T rows.
    std::span<std::uint32_t> in = levels.work[1], even = levels.work[2], next = levels.work[3];
    std::fill(in.begin(), in.end(), 0);
    for (std::size_t i = 0; i < m; ++i) in[2 * i] = m - 1 - i < f.size() ? f[m - 1 - i] : 0;
    for (int s = levels.lg - 1; s >= 0; --s) {
        const std::size_t stride = m >> s, rows = std::size_t(1) << s;  // outputs: rows [rows, 2 rows), x < stride / 2
        t.forward(in);
        t.inverse_product(in, levels.even(s), even, Half::kUpper);
        t.inverse_product(in, levels.odd(s), in, Half::kUpper);  // odd outputs in place
        if (s == 0) {
            for (std::size_t k = 0; k < n; ++k) h[k] = (k % 2 ? in : even)[m + k / 2];
            return;
        }
        // Next level: stride 2 stride, row i = even and odd outputs of row rows + i interleaved.
        std::fill(next.begin(), next.end(), 0);
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < stride / 2; ++j) {
                next[2 * stride * i + 2 * j] = even[stride * (rows + i) + j];
                next[2 * stride * i + 2 * j + 1] = in[stride * (rows + i) + j];
            }
        std::swap(in, next);
    }
}

}  // namespace poly
