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

// Block 2 of 4: the residual sum W_2 q_0 + W_1 q_1 and block 3's W_3 q_0 + W_2 q_1 by three leaf
// products instead of four (the 2 x 2 Toeplitz product):
//   m1 = W_2 (q_0 + q_1),  m2 = (W_1 - W_2) q_1,  m3 = (W_3 - W_2) q_0,
//   W_2 q_0 + W_1 q_1 = m1 + m2,  W_3 q_0 + W_2 q_1 = m1 + m3.
// The leaves of m1 + m2 go through the inverse butterflies; those of m1 + m3 (< 2P, with the
// factor 2^-32 of leaf products) replace W_3's. All operands are transforms of the same length;
// the output may be q_1.
struct ToeplitzBottom {
    static constexpr bool kForward = false, kInverse = true;
    const std::uint32_t *roots, *inverse_roots, *w1, *w2, *q0, *q1;
    std::uint32_t* w3;  // W_3's transform, then the leaves of m1 + m3

    // Windows of q_0 + q_1, W_1 - W_2 and W_3 - W_2 for group g.
    [[gnu::always_inline]] void prepare(std::size_t g, Window (&window)[3][4]) const {
        Vec sum[4], d1[4], d3[4];
#pragma GCC unroll 4
        for (std::size_t t = 0; t < 4; ++t) {
            const std::size_t leaf = 8 * (4 * g + t);
            const Vec v2 = load(w2 + leaf);
            sum[t] = reduce(add(load(q0 + leaf), load(q1 + leaf)), kP);
            d1[t] = difference(load(w1 + leaf), v2);
            d3[t] = difference(load(w3 + leaf), v2);
        }
        fill_windows(window[0], sum, roots, g);
        fill_windows(window[1], d1, roots, g);
        fill_windows(window[2], d3, roots, g);
    }

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        Window window[2][3][4];
        prepare(first, window[0]);
        for (std::size_t j = 0; j < count; ++j, out += 32) {
            if (j + 1 < count) prepare(first + j + 1, window[(j + 1) & 1]);
            const std::size_t g = first + j;
            const auto& w = window[j & 1];
            Vec f[4];
#pragma GCC unroll 4
            for (std::size_t t = 0; t < 4; ++t) {
                const std::size_t leaf = 8 * (4 * g + t);
                const Vec m1 = leaf_product(w[0][t], w2 + leaf);
                f[t] = low(add(m1, leaf_product(w[1][t], q1 + leaf)));
                store(w3 + leaf, low(add(m1, leaf_product(w[2][t], q0 + leaf))));
            }
            inverse_h1(f, Group(inverse_roots, g));
#pragma GCC unroll 4
            for (std::size_t t = 0; t < 4; ++t) store(out + 8 * t, f[t]);
        }
    }
};

// Block 3 of 4: the residual sum (m1 + m3) + W_1 q_2, the first term as ToeplitzBottom left it.
// The output may be q_2.
struct SideProductBottom {
    static constexpr bool kForward = false, kInverse = true;
    InverseProductBottom term;  // W_1 (a) and q_2 (b)
    const std::uint32_t* side;

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        Window window[2][4];
        term.prepare(first, window[0]);
        for (std::size_t j = 0; j < count; ++j, out += 32) {
            if (j + 1 < count) term.prepare(first + j + 1, window[(j + 1) & 1]);
            const std::size_t g = first + j;
            Vec f[4];
#pragma GCC unroll 4
            for (std::size_t t = 0; t < 4; ++t) {
                const std::size_t leaf = 8 * (4 * g + t);
                f[t] = low(add(leaf_product(window[j & 1][t], term.product.b + leaf), load(side + leaf)));
            }
            inverse_h1(f, Group(term.product.inverse_roots, g));
#pragma GCC unroll 4
            for (std::size_t t = 0; t < 4; ++t) store(out + 8 * t, f[t]);
        }
    }
};

// A log_derivative callback for the blocks' transforms that ignores them.
struct IgnoreTransforms {
    void operator()(std::size_t, std::span<const std::uint32_t>) const {}
};

// Buffers of length 2k log_derivative uses: T(h), W_1 .. W_(B-1), q_0's, and one for the later
// blocks; at least 3 (for B = 1: the inverse's scratch, 2 buffers, follows T(h)).
inline std::size_t log_buffers(std::size_t n) { return std::max<std::size_t>(log_blocks(n) + 2, 3); }

// q = f'/f mod x^(n-1) for n >= 2, f[0] != 0, in blocks q_j of k coefficients (k = log_block(n))
// from h = 1 / f mod x^k, with transforms of length 2k: for d = f' and Q = q mod x^(jk),
// (d - f Q) is divisible by x^(jk), and
//   q_j = h (d - f Q)[jk, (j+1)k) mod x^k,
//   (f Q)[jk, (j+1)k) = sum_(i<j) (W_(j-i) q_i)[k, 2k),  W_t = f[(t-1)k, (t+1)k).
// The sum is one inverse transform of the products of the stored transforms of W_t and q_i
// (for 4 blocks, those of blocks 2 and 3 by ToeplitzBottom and SideProductBottom).
// Cost for 4 blocks, the inverse to k included: 23 transforms of length 2k and 11 leaf products.
// Blocks j >= 1 share one buffer: the residual (over T(q_(j-1)) for j >= 2), q_j, T(q_j).
// sink(first, q) receives q[first, first + q.size()), block by block, as an aligned span
// readable to the next multiple of 8. After it returns, f is read only at indices
// > first + q.size(). transformed(j, T(q_j)) receives the transform of length 2k of each block
// but the last, valid until the next block starts (T(q_0) until the end).
// scratch: log_derivative_scratch(n) words; t: lg_max >= log_derivative_log(n).
template <class Sink, class Transformed = IgnoreTransforms>
[[gnu::always_inline]] inline void log_derivative(const Transform& t, std::span<const std::uint32_t> f, std::size_t n,
                                                  std::span<std::uint32_t> scratch, const Sink& sink,
                                                  const Transformed& transformed = {}) {
    const std::size_t k = log_block(n), len = 2 * k, blocks = log_blocks(n);
    const auto buffer = [&scratch, len](std::size_t i) { return scratch.subspan(i * Arena::footprint(len), len); };
    const std::span<std::uint32_t> ht = buffer(0), q0 = buffer(blocks);
    const auto window = [&buffer](std::size_t t) { return buffer(t); };

    inverse(t, f, ht.first(k), scratch.subspan(Arena::footprint(len), inverse_scratch(k)));
    t.forward(ht.first(k), 0, ht);
    for (std::size_t s = 1; s < blocks; ++s) {
        const std::size_t from = std::min(f.size(), (s - 1) * k);
        t.forward(f.subspan(from, std::min(f.size() - from, len)), 0, window(s));
    }
    for (std::size_t j = 0; j < blocks; ++j) {
        const std::size_t first = j * k, count = std::min(k, n - 1 - first);
        const std::span<std::uint32_t> work = j == 0 ? q0 : buffer(blocks + 1);
        if (j == 0) {
            derivative(f.first(std::min(f.size(), count + 1)), work.first(count));
        } else {
            if (j == 1) {
                t.inverse_product(window(1), q0, work, Half::kUpper);
            } else if (blocks == 3) {
                const Transform::Pair pairs[2] = {{window(2), q0}, {window(1), work}};
                t.inverse_product_sum(pairs, work, Half::kUpper);
            } else if (j == 2) {
                t.inverse_with(ToeplitzBottom{t.roots(), t.inverse_roots(), window(1).data(), window(2).data(), q0.data(),
                                              work.data(), window(3).data()},
                               work, Half::kUpper);
            } else {
                const InverseProductBottom product{{t.roots(), t.inverse_roots(), work.data()}, window(1).data()};
                t.inverse_with(SideProductBottom{product, window(3).data()}, work, Half::kUpper);
            }
            subtract_from_derivative(f, first, count, work.data(), k);
        }
        t.cyclic_product(work.first(count), 0, work, ht, Half::kLower);
        sink(first, std::span<const std::uint32_t>(work.first(count)));
        if (j + 1 < blocks) {
            t.forward(work.first(k), 0, work);
            transformed(j, std::span<const std::uint32_t>(work));
        }
    }
}

inline int log_derivative_log(std::size_t n) { return std::countr_zero(2 * log_block(n)); }

inline std::size_t log_derivative_scratch(std::size_t n) { return log_buffers(n) * Arena::footprint(2 * log_block(n)); }

}  // namespace detail

// Transform length log uses for n coefficients: the Transform needs lg_max >= this.
inline int log_log(std::size_t n) {
    return n <= detail::kLogBase ? Transform::kMinLog : std::countr_zero(2 * detail::log_block(n));
}

// Scratch words for log() of n coefficients.
inline std::size_t log_scratch(std::size_t n) {
    return n <= detail::kLogBase ? 0 : detail::log_buffers(n) * Arena::footprint(2 * detail::log_block(n));
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
