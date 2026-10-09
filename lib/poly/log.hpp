// Logarithm of a power series modulo 998244353: g = log(f) mod x^n, f[0] = 1.
// Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::log_log(n)) + poly::log_scratch(n) + ...);
//   poly::Transform t(arena, poly::log_log(n));
//   poly::log(t, f, g, arena.take(poly::log_scratch(n)));  // g.size() == n; g may be f
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

// q = f'/f is computed in at most this many blocks.
inline constexpr std::size_t kLogBlocks = 4;

// g = log(f) mod x^n, n = g.size() <= kLogBase, by i g_i = i f_i - sum_(0<k<i) k g_k f_(i-k).
inline void log_direct(std::span<const std::uint32_t> f, std::span<std::uint32_t> g) {
    using ntt::detail::multiply_mod;
    std::uint32_t a[kLogBase] = {}, inv[kLogBase] = {0, 1}, kg[kLogBase] = {};  // a = f, inv[i] = 1 / i, kg[k] = k g_k
    std::copy_n(f.begin(), std::min(f.size(), g.size()), a);
    for (std::size_t i = 2; i < g.size(); ++i) inv[i] = multiply_mod(kP - kP / std::uint32_t(i), inv[kP % i]);
    g[0] = 0;
    for (std::size_t i = 1; i < g.size(); ++i) {
        std::uint64_t sum = multiply_mod(std::uint32_t(i), a[i]);  // fewer than kLogBase terms <= P
        for (std::size_t k = 1; k < i; ++k) sum += kP - std::uint64_t(kg[k]) * a[i - k] % kP;
        kg[i] = std::uint32_t(sum % kP);
        g[i] = multiply_mod(kg[i], inv[i]);
    }
}

// Coefficients per block of q = f'/f (n - 1 of them), a power of two >= 32.
inline std::size_t log_block(std::size_t n) {
    return std::max<std::size_t>(32, std::bit_ceil((n - 1 + kLogBlocks - 1) / kLogBlocks));
}

inline std::size_t log_blocks(std::size_t n) { return (n - 1 + log_block(n) - 1) / log_block(n); }

// out[c] = d[first + c] - out[size + c] for c < count, d = f' (coefficients of f past f.size()
// are zero). out must not overlap f.
inline void subtract_from_derivative(std::span<const std::uint32_t> f, std::size_t first, std::size_t count,
                                     std::uint32_t* out, std::size_t size) {
    const std::size_t inside = f.size() > first + 1 ? std::min(count, f.size() - first - 1) : 0, full = inside / 8 * 8;
    Indices index(first + 1);
    for (std::size_t c = 0; c < full; c += 8, index.next()) {
        const Vec d = canonical(montgomery(load_unaligned(f.data() + first + 1 + c), index.value()));
        store_unaligned(out + c, reduce(_mm256_sub_epi32(add(d, broadcast(kP)), load_unaligned(out + size + c)), kP));
    }
    for (std::size_t c = full; c < count; ++c) {
        const std::uint32_t d = c < inside ? ntt::detail::multiply_mod(std::uint32_t(first + 1 + c), f[first + 1 + c]) : 0;
        out[c] = (d + kP - out[size + c]) % kP;
    }
}

// q = f'/f mod x^(n-1) for n >= 2, f[0] != 0, in blocks q_j of k coefficients (k = log_block(n))
// from h = 1 / f mod x^k, with transforms of length 2k: for d = f' and Q = q mod x^(jk),
// (d - f Q) is divisible by x^(jk), and
//   q_j = h (d - f Q)[jk, (j+1)k) mod x^k,
//   (f Q)[jk, (j+1)k) = sum_(i<j) (W_(j-i) q_i)[k, 2k),  W_t = f[(t-1)k, (t+1)k).
// The sum is one inverse transform of the products of the stored transforms of W_t and q_i.
// Cost for 4 blocks: the inverse to k, then 23 transforms of length 2k and 12 leaf products.
// sink(first, q) receives q[first, first + q.size()), block by block, as an aligned span
// readable to the next multiple of 8. After it returns, f is read only at indices
// > first + q.size(). scratch: log_derivative_scratch(n) words; t: lg_max >= log_derivative_log(n).
template <class Sink>
[[gnu::always_inline]] inline void log_derivative(const Transform& t, std::span<const std::uint32_t> f, std::size_t n,
                                                  std::span<std::uint32_t> scratch, const Sink& sink) {
    const std::size_t k = log_block(n), len = 2 * k, blocks = log_blocks(n);
    // Buffers: the transforms of h, W_1 .. W_(B-1), q_0 .. q_(B-2), and work space.
    const auto buffer = [&scratch, len](std::size_t i) { return scratch.subspan(i * Arena::footprint(len), len); };
    const std::span<std::uint32_t> ht = buffer(0), work = buffer(2 * blocks - 1);
    const auto window = [&buffer](std::size_t t) { return buffer(t); };
    const auto q_transform = [&buffer, blocks](std::size_t i) { return buffer(blocks + i); };

    inverse(t, f, ht.first(k), scratch.subspan(Arena::footprint(len), inverse_scratch(k)));
    t.forward(ht.first(k), 0, ht);
    for (std::size_t s = 1; s < blocks; ++s) {
        const std::size_t from = std::min(f.size(), (s - 1) * k);
        t.forward(f.subspan(from, std::min(f.size() - from, len)), 0, window(s));
    }
    for (std::size_t j = 0; j < blocks; ++j) {
        const std::size_t first = j * k, count = std::min(k, n - 1 - first);
        if (j == 0) {
            derivative(f.first(std::min(f.size(), count + 1)), work.first(count));
        } else {
            Transform::Pair pairs[kLogBlocks - 1];
            for (std::size_t i = 0; i < j; ++i) pairs[i] = {window(j - i), q_transform(i)};
            t.inverse_product_sum(std::span(pairs, j), work, Half::kUpper);
            subtract_from_derivative(f, first, count, work.data(), k);
        }
        t.cyclic_product(work.first(count), 0, work, ht, Half::kLower);
        if (j + 1 < blocks) t.forward(work.first(k), 0, q_transform(j));
        sink(first, std::span<const std::uint32_t>(work.first(count)));
    }
}

inline int log_derivative_log(std::size_t n) { return std::countr_zero(2 * log_block(n)); }

// 2 B buffers, at least 3 (for B = 1: the inverse's scratch, 2 buffers, follows T(h)).
inline std::size_t log_derivative_scratch(std::size_t n) {
    return std::max<std::size_t>(2 * log_blocks(n), 3) * Arena::footprint(2 * log_block(n));
}

}  // namespace detail

// Transform length log uses for n coefficients: the Transform needs lg_max >= this.
inline int log_log(std::size_t n) {
    return n <= detail::kLogBase ? Transform::kMinLog : std::countr_zero(2 * detail::log_block(n));
}

// Scratch words for log() of n coefficients.
inline std::size_t log_scratch(std::size_t n) {
    return n <= detail::kLogBase ? 0 : 2 * detail::log_blocks(n) * Arena::footprint(2 * detail::log_block(n));
}

// g = log(f) mod x^n for n = g.size() >= 1. f[0] = 1; coefficients of f past f.size() are zero.
// g may be f; otherwise the two must not overlap. scratch: log_scratch(n) words, 32-byte aligned
// (from an Arena). t: lg_max >= log_log(n).
//
// log f is the integral of q = f'/f mod x^(n-1), computed in up to 4 blocks by
// detail::log_derivative. In place, block q[first, first + c) is integrated into
// g[first + 1, first + c + 1), which log_derivative no longer reads.
inline void log(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size();
    if (n <= kLogBase) return log_direct(f, g);
    log_derivative(t, f, n, scratch, [g](std::size_t first, std::span<const std::uint32_t> q) {
        detail::divide_by_index(first + 1, g.subspan(first + 1, q.size()), [q](std::size_t i) { return load(q.data() + i); });
    });
    g[0] = 0;
}

}  // namespace poly
