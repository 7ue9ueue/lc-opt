// Polynomial interpolation modulo 998244353 on the product tree of evaluation.hpp. x86-64 with
// AVX2. Design: lib/poly/notes.md (Interpolation).
//
//   poly::Arena arena(poly::interpolate_words(m));
//   poly::interpolate(arena, points, values, c);  // c: the m coefficients of f, f(points[i]) = values[i]
//
// For m distinct points a_i, M = prod (x - a_i) and Q = prod (1 - a_i x) = x^m M(1/x):
// f = sum_i w_i M / (x - a_i) with the weights w_i = y_i / M'(a_i), and f's coefficients reversed
// are those of R = sum_i w_i Q / (1 - a_i x).
// - M'(a_i): PointTree's descent as in evaluate() for M', from the root's window rev(M') / Q mod
//   x^m8 (divide()).
// - R by the transpose of the descent: node v's sum R_v = sum over its points of w_i Q_v / (1 - a_i x)
//   is R_l Q_r + R_r Q_l, of degree < s_v: exact in products of the stored transforms' length L.
//   Per node: the products and their sum, then an inverse and a forward_upper of length L (the
//   doubling to the parent's length), as ProductTree's nodes.
// - Lane tree: the descent and the sums in one depth-first pass (InterpolationTree::lanes), so a
//   node's sum is built right after its children's descents. Its base (BaseInterpolation) runs the
//   descent to the slots, inverts the values (Montgomery's batch inversion), and builds the sums
//   from the same schoolbook products.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/evaluation.hpp"

namespace poly {

namespace detail {

// h 2^32 + l -> h (2^32 mod P) + l < 2^62 + 2^32, in 64-bit lanes.
inline Vec fold(Vec s) {
    return _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(s, 32), broadcast(kR)), _mm256_blend_epi32(s, _mm256_setzero_si256(), 0xAA));
}

// The canonical words of the K sums (even[k], odd[k]) / 2^32 of terms products < P^2 in all, at
// most 16 per sum (< 2^64): up to 17 terms as montgomery_sums; more: each sum folded (together
// < 2^63 + 2^33), then the Montgomery step alone (< 2^64).
template <std::size_t K>
[[gnu::always_inline]] inline Vec cross_sum(const Vec (&even)[K], const Vec (&odd)[K], std::size_t terms) {
    if constexpr (K == 1) {
        return montgomery_sums(even[0], odd[0], terms);
    } else {
        if (terms <= 17) return montgomery_sums(_mm256_add_epi64(even[0], even[1]), _mm256_add_epi64(odd[0], odd[1]), terms);
        return montgomery_sums(_mm256_add_epi64(fold(even[0]), fold(even[1])), _mm256_add_epi64(fold(odd[0]), fold(odd[1])), 0);
    }
}

// Lanes: out[t] = (sum_i a[i] d[t - i] + sum_i b[i] c[t - i]) / 2^32 for t < n + k, a of n vectors,
// d of k + 1, b of k, c of n + 1, all canonical; out canonical. n, k <= 16. For
// R_v = R_l Q_r + R_r Q_l: a = R_l, b = R_r, c = Q_l, d = Q_r.
[[gnu::noinline]] inline void cross_lanes(const std::uint32_t* a, std::size_t n, const std::uint32_t* b, std::size_t k,
                                          const std::uint32_t* c, const std::uint32_t* d, std::uint32_t* out) {
    for (std::size_t t = 0; t < n + k; ++t) {
        Vec even[2] = {}, odd[2] = {};
        const std::size_t first[2] = {t > k ? t - k : 0, t > n ? t - n : 0}, last[2] = {std::min(t, n - 1), std::min(t, k - 1)};
        for (std::size_t i = first[0]; i <= last[0]; ++i) {
            const Vec x = load(a + 8 * i), y = load(d + 8 * (t - i));
            even[0] = _mm256_add_epi64(even[0], _mm256_mul_epu32(x, y));
            odd[0] = _mm256_add_epi64(odd[0], _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32)));
        }
        for (std::size_t i = first[1]; i <= last[1]; ++i) {
            const Vec x = load(b + 8 * i), y = load(c + 8 * (t - i));
            even[1] = _mm256_add_epi64(even[1], _mm256_mul_epu32(x, y));
            odd[1] = _mm256_add_epi64(odd[1], _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32)));
        }
        store(out + 8 * t, cross_sum(even, odd, (last[0] + 1 - first[0]) + (last[1] + 1 - first[1])));
    }
}

// The same for n = k = D in blocks of B outputs, their sums in registers (one per output, two
// when 2D > 17 terms); the odd lanes shifted once, and a, b padded with B - 1 zero vectors on
// each side so that every term of a block is in range (the padding adds zero products).
template <std::size_t D, std::size_t B>
[[gnu::always_inline]] inline void cross_lanes(const std::uint32_t* a, const std::uint32_t* b, const std::uint32_t* c,
                                               const std::uint32_t* d, std::uint32_t* out) {
    constexpr std::size_t kPad = B - 1, kSums = 2 * D > 17 ? 2 : 1;
    Vec x[D + 2 * kPad], xo[D + 2 * kPad], y[D + 2 * kPad], yo[D + 2 * kPad], co[D + 1], dq[D + 1];
    for (std::size_t i = 0; i < kPad; ++i)
        x[i] = xo[i] = y[i] = yo[i] = x[D + kPad + i] = xo[D + kPad + i] = y[D + kPad + i] = yo[D + kPad + i] = _mm256_setzero_si256();
#pragma GCC unroll 16
    for (std::size_t i = 0; i < D; ++i) {
        x[kPad + i] = load(a + 8 * i), xo[kPad + i] = _mm256_srli_epi64(x[kPad + i], 32);
        y[kPad + i] = load(b + 8 * i), yo[kPad + i] = _mm256_srli_epi64(y[kPad + i], 32);
    }
#pragma GCC unroll 17
    for (std::size_t j = 0; j <= D; ++j) co[j] = _mm256_srli_epi64(load(c + 8 * j), 32), dq[j] = _mm256_srli_epi64(load(d + 8 * j), 32);
    for (std::size_t t0 = 0; t0 < 2 * D; t0 += B) {
        Vec even[B][kSums] = {}, odd[B][kSums] = {};
        const std::size_t first = t0 + 1 > D ? t0 + 1 - D : 0, last = std::min(D, t0 + B - 1);
        for (std::size_t j = first; j <= last; ++j) {
            const Vec qd = load(d + 8 * j), qc = load(c + 8 * j);
#pragma GCC unroll 8
            for (std::size_t s = 0; s < B; ++s) {
                const std::size_t i = kPad + t0 + s - j;
                even[s][0] = _mm256_add_epi64(even[s][0], _mm256_mul_epu32(x[i], qd));
                odd[s][0] = _mm256_add_epi64(odd[s][0], _mm256_mul_epu32(xo[i], dq[j]));
                even[s][kSums - 1] = _mm256_add_epi64(even[s][kSums - 1], _mm256_mul_epu32(y[i], qc));
                odd[s][kSums - 1] = _mm256_add_epi64(odd[s][kSums - 1], _mm256_mul_epu32(yo[i], co[j]));
            }
        }
#pragma GCC unroll 8
        for (std::size_t s = 0; s < B; ++s) {
            const std::size_t t = t0 + s, terms = 2 * (std::min(t, D - 1) + 1 - (t > D ? t - D : 0));
            if (t < 2 * D) store(out + 8 * t, cross_sum(even[s], odd[s], terms));
        }
    }
}

// The same for n = k = D <= 8, unrolled: at most 16 terms per output, one sum.
template <std::size_t D>
[[gnu::always_inline]] inline void cross_lanes(const std::uint32_t* a, const std::uint32_t* b, const std::uint32_t* c,
                                               const std::uint32_t* d, std::uint32_t* out) {
    static_assert(2 * D <= 17);
    Vec x[D], xo[D], y[D], yo[D], p[D + 1], po[D + 1], q[D + 1], qo[D + 1];
#pragma GCC unroll 8
    for (std::size_t i = 0; i < D; ++i) {
        x[i] = load(a + 8 * i), xo[i] = _mm256_srli_epi64(x[i], 32);
        y[i] = load(b + 8 * i), yo[i] = _mm256_srli_epi64(y[i], 32);
    }
#pragma GCC unroll 9
    for (std::size_t j = 0; j <= D; ++j) {
        p[j] = load(c + 8 * j), po[j] = _mm256_srli_epi64(p[j], 32);
        q[j] = load(d + 8 * j), qo[j] = _mm256_srli_epi64(q[j], 32);
    }
#pragma GCC unroll 16
    for (std::size_t t = 0; t < 2 * D; ++t) {
        const std::size_t first = t > D ? t - D : 0, last = std::min(t, D - 1);
        Vec even = _mm256_setzero_si256(), odd = _mm256_setzero_si256();
#pragma GCC unroll 8
        for (std::size_t i = first; i <= last; ++i) {
            even = _mm256_add_epi64(even, _mm256_add_epi64(_mm256_mul_epu32(x[i], q[t - i]), _mm256_mul_epu32(y[i], p[t - i])));
            odd = _mm256_add_epi64(odd, _mm256_add_epi64(_mm256_mul_epu32(xo[i], qo[t - i]), _mm256_mul_epu32(yo[i], po[t - i])));
        }
        store(out + 8 * t, montgomery_sums(even, odd, 2 * (last + 1 - first)));
    }
}

// Lanes, one level of a perfect tree: the sums of nodes of degree 2D from pairs (2i, 2i + 1) of
// nodes of degree D, their sums (D vectors each) and products (D + 1 vectors each).
template <std::size_t D>
void cross_level(const std::uint32_t* sums, const std::uint32_t* products, std::uint32_t* out, std::size_t nodes) {
    for (std::size_t i = 0; i < nodes; ++i) {
        const std::uint32_t *r = sums + 8 * D * (2 * i), *q = products + 8 * (D + 1) * (2 * i);
        if constexpr (D <= 8) cross_lanes<D>(r, r + 8 * D, q, q + 8 * (D + 1), out + 16 * D * i);
        else cross_lanes<D, 2>(r, r + 8 * D, q, q + 8 * (D + 1), out + 16 * D * i);
    }
}

// Lanes: 1 / x in Montgomery form for x in Montgomery form (< 2P, nonzero mod P), as x^(P - 2).
inline Vec inverse_montgomery(Vec x) {
    Vec result = broadcast(kR), power = x;  // kR = 2^32 mod P: 1 in Montgomery form
    for (std::uint32_t e = kP - 2; e; e >>= 1) {
        if (e & 1) result = montgomery(result, power);
        power = montgomery(power, power);
    }
    return result;
}

// The interpolation below a lane node of degree <= LaneLayout::kBase, slots [lo, hi), from the
// node's window: the descent to the slots as BaseDescent, whose windows v_i (Montgomery form) give
// the weights y_i / v_i; then the node's sum of w_i prod_(j != i) (1 - a_j x) over its slots by
// the same products. A power of two of slots runs level by level; other counts by recursion.
class BaseInterpolation {
public:
    // y: the values, one per point (m of them), zero up to a multiple of 8.
    BaseInterpolation(const PointSlots& slots, const std::uint32_t* y, std::size_t m) : slots_(slots), y_(y), m_(m) {}

    // The sum (hi - lo vectors, canonical) from the window w (hi - lo vectors); valid until the
    // next call.
    const std::uint32_t* run(std::size_t lo, std::size_t hi, const std::uint32_t* w) {
        const std::size_t count = hi - lo;
        lo_ = lo;
        if (count == 1) {
            store(values_, load(w));
            weights(1);
            return values_;
        }
        if (std::has_single_bit(count)) return run_levels(count, w);
        free_ = memory_;
        build(1, lo, hi);
        descend(1, lo, hi, w);
        weights(count);
        ascend(1, lo, hi, sums_);
        return sums_;
    }

private:
    // count = 2^k slots: products and windows as BaseDescent::run_levels, the slots' windows into
    // values_; then the sums of degree 2, .. count, level by level.
    const std::uint32_t* run_levels(std::size_t count, const std::uint32_t* w) {
        const std::uint32_t* products[6];
        std::uint32_t* at = memory_;
        for (std::size_t k = 0; k < count; ++k) slots_.load(lo_ + k, at + 16 * k);
        products[0] = at;
        at += 16 * count;
        for (std::size_t d = 1, j = 1; 2 * d < count; d *= 2, ++j) {
            with_degree(d, [&]<std::size_t D>() {
                if constexpr (D <= LaneLayout::kBase / 4) product_level<D>(products[j - 1], at, count / (2 * D));
            });
            products[j] = at;
            at += 8 * (2 * d + 1) * (count / (2 * d));
        }
        std::uint32_t* const buffers[2] = {at, at + 8 * count};
        const std::uint32_t* parents = w;
        for (std::size_t d = count / 2, j = std::countr_zero(count) - 1;; d /= 2, --j) {
            std::uint32_t* const out = d == 1 ? values_ : buffers[j & 1];
            with_degree(d, [&]<std::size_t D>() { middle_level<D>(parents, products[j], out, count / D); });
            if (d == 1) break;
            parents = out;
        }
        weights(count);
        const std::uint32_t* sums = values_;
        for (std::size_t d = 1, j = 0; d < count; d *= 2, ++j) {
            std::uint32_t* const out = 2 * d == count ? sums_ : buffers[j & 1];
            with_degree(d, [&]<std::size_t D>() { cross_level<D>(sums, products[j], out, count / (2 * D)); });
            sums = out;
        }
        return sums_;
    }

    // The products of node v = [lo, hi) and of its descendants, heap order, as BaseDescent::build.
    const std::uint32_t* build(std::size_t v, std::size_t lo, std::size_t hi) {
        std::uint32_t* const c = free_;
        if (hi - lo == 1) {
            slots_.load(lo, c);
            free_ += 16;
            return c;
        }
        const std::size_t mid = lo + (hi - lo) / 2;
        product_[2 * v] = build(2 * v, lo, mid);
        product_[2 * v + 1] = build(2 * v + 1, mid, hi);
        if (v == 1) return nullptr;
        std::uint32_t* const out = free_;
        multiply_lanes_any(product_[2 * v], mid - lo, product_[2 * v + 1], hi - mid, out);
        free_ += 8 * (hi - lo + 1);
        return out;
    }

    // The windows down to the slots, into values_.
    void descend(std::size_t v, std::size_t lo, std::size_t hi, const std::uint32_t* w) {
        if (hi - lo == 1) return store(values_ + 8 * (lo - lo_), load(w));
        const std::size_t mid = lo + (hi - lo) / 2;
        std::uint32_t* const left = free_;
        std::uint32_t* const right = left + 8 * (mid - lo);
        free_ = right + 8 * (hi - mid);
        middle_lanes(w, hi - mid, product_[2 * v + 1], mid - lo, left);
        middle_lanes(w, mid - lo, product_[2 * v], hi - mid, right);
        descend(2 * v, lo, mid, left);
        descend(2 * v + 1, mid, hi, right);
        free_ = left;
    }

    // The sum of node v = [lo, hi) into out (hi - lo vectors) from the weights in values_.
    void ascend(std::size_t v, std::size_t lo, std::size_t hi, std::uint32_t* out) {
        if (hi - lo == 1) return store(out, load(values_ + 8 * (lo - lo_)));
        const std::size_t mid = lo + (hi - lo) / 2;
        std::uint32_t* const left = free_;
        std::uint32_t* const right = left + 8 * (mid - lo);
        free_ = right + 8 * (hi - mid);
        ascend(2 * v, lo, mid, left);
        ascend(2 * v + 1, mid, hi, right);
        cross_lanes(left, mid - lo, right, hi - mid, product_[2 * v], product_[2 * v + 1], out);
        free_ = left;
    }

    // In place: values_[k] (Montgomery form, slot lo_ + k) becomes y / value (canonical), by one
    // inversion (all Montgomery form) in 4 chains, chain c over k = c mod 4: prefix products
    // p_k = p_(k-4) x_k (x = 1 past count), e_c = 1 / (chain c's product) from the inverse of the
    // 4 products' product, then 1 / x_k = e_c p_(k-4) and e_c <- e_c x_k from the top. Lanes of
    // padding points: value 1, weight 0.
    void weights(std::size_t count) {
        constexpr std::size_t kChains = 4;
        const Vec iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), one = broadcast(kR);
        const std::size_t groups = (count + kChains - 1) / kChains;
        Vec prefix[kMaxSlots];
        for (std::size_t g = 0; g < groups; ++g) {
#pragma GCC unroll 4
            for (std::size_t c = 0; c < kChains; ++c) {
                const std::size_t k = kChains * g + c, first = 8 * (lo_ + k);
                Vec x = one;
                if (k < count) {
                    x = load(values_ + 8 * k);
                    if (first + 8 > m_) {
                        x = _mm256_blendv_epi8(one, x, _mm256_cmpgt_epi32(broadcast(std::uint32_t(m_ - first)), iota));
                        store(values_ + 8 * k, x);
                    }
                }
                prefix[k] = g ? montgomery(prefix[k - kChains], x) : x;
            }
        }
        const Vec* const last = prefix + kChains * (groups - 1);
        const Vec p01 = montgomery(last[0], last[1]), p23 = montgomery(last[2], last[3]);
        const Vec inverse = inverse_montgomery(montgomery(p01, p23));
        const Vec i01 = montgomery(inverse, p23), i23 = montgomery(inverse, p01);
        Vec e[kChains] = {montgomery(i01, last[1]), montgomery(i01, last[0]), montgomery(i23, last[3]), montgomery(i23, last[2])};
        for (std::size_t g = groups; g-- > 0;) {
#pragma GCC unroll 4
            for (std::size_t c = 0; c < kChains; ++c) {
                const std::size_t k = kChains * g + c;
                if (k >= count) continue;
                const Vec x = load(values_ + 8 * k), reciprocal = g ? montgomery(e[c], prefix[k - kChains]) : e[c];
                e[c] = montgomery(e[c], x);
                store(values_ + 8 * k, canonical(montgomery(reciprocal, load(y_ + 8 * (lo_ + k)))));
            }
        }
    }

    static constexpr std::size_t kMaxSlots = LaneLayout::kBase;

    // Products: as BaseDescent's; windows or sums: at most 2 * 32 vectors.
    alignas(32) std::uint32_t memory_[8 * 320];
    alignas(32) std::uint32_t values_[8 * kMaxSlots];  // windows of the slots, then their weights
    alignas(32) std::uint32_t sums_[8 * kMaxSlots];
    const std::uint32_t* product_[64];
    std::uint32_t* free_ = memory_;
    const PointSlots& slots_;
    const std::uint32_t* y_;
    std::size_t m_, lo_ = 0;
};

}  // namespace detail

// A PointTree that also runs the interpolation's passes.
class InterpolationTree : public PointTree {
public:
    InterpolationTree(Arena& arena, std::span<const std::uint32_t> points) : PointTree(arena, points), m_(points.size()) {}

    // Scratch words interpolate() takes for m points.
    static std::size_t scratch_words(std::size_t m) {
        const std::size_t m8 = (m + 7) / 8 * 8, lane_length = LaneLayout::length(std::uint32_t(m8 / 8));
        const std::size_t stride = std::max<std::size_t>(8, lane_length), top_length = StandardLayout::length(std::uint32_t(m8));
        return descend_words(m) + 10 * Arena::footprint(8 * stride) + 6 * Arena::footprint(top_length) + 4096;
    }

    // r = sum_(i < m) (y_i / v_i) prod_(j != i) (1 - a_j x) (canonical; r has
    // StandardLayout::length(size()) words, the first size() the coefficients) for the descent's
    // values v_i = [x^(m8 - 1)] w(x) prod_(j != i) (1 - a_j x), which must be nonzero; w (canonical)
    // has m8 = size() coefficients and may share r's memory, y m8 words (32-byte aligned, zero
    // past m). scratch: scratch_words(m) words from an Arena.
    void interpolate(std::span<const std::uint32_t> w, std::span<const std::uint32_t> y, std::span<std::uint32_t> r,
                     std::span<std::uint32_t> scratch) {
        using namespace detail;
        Stack stack(scratch);
        const std::size_t lane_length = LaneLayout::length(std::uint32_t(count_)), stride = std::max<std::size_t>(8, lane_length);
        const LaneState root = lane_state(w, kR, stack);  // the values in Montgomery form
        BaseInterpolation base(slots_, y.data(), m_);
        std::uint32_t* const sums = stack.take(8 * stride);  // the lane products' sums, lanes layout
        if (count_ <= LaneLayout::kBase) {
            std::copy_n(base.run(0, count_, root.state + 8 * (lane_length - count_)), 8 * count_, sums);
        } else {
            lanes(0, count_, root.state, root.u, stack, base, sums, lane_length);
            t_.inverse({sums, 8 * lane_length}, {sums, 8 * lane_length});
        }
        std::uint32_t* const columns = root.state;  // lane l's sum (count_ coefficients) at l stride
        for (std::size_t j = 0; j < count_; j += 8) {
            Vec v[8];
            for (std::size_t i = 0; i < 8; ++i) v[i] = load(sums + 8 * (j + i));
            transpose8(v);
            for (std::size_t l = 0; l < 8; ++l) store(columns + l * stride + j, v[l]);
        }
        const std::size_t length = StandardLayout::length(std::uint32_t(m8_));
        top(0, 8, columns, stride, stack, r.data(), length);
        t_.inverse(r.first(length), r.first(length));
    }

private:
    // Node [lo, hi) of the lane tree, of degree > kBase, with state s and u = 1 / f_v as in
    // descend_lanes: the descent to its slots, then the transform of its sum of length
    // out_length >= its length into out (8 out_length words).
    void lanes(std::size_t lo, std::size_t hi, const std::uint32_t* s, std::uint32_t u, detail::Stack& stack,
               detail::BaseInterpolation& base, std::uint32_t* out, std::size_t out_length) {
        using namespace detail;
        const LaneTree& tree = *lanes_;
        const std::size_t length = LaneLayout::length(std::uint32_t(tree.degree(lo, hi))), words = 8 * length;
        const std::size_t mid = tree.split(lo, hi), mark = stack.mark();
        const std::uint32_t* const stored[2] = {lane_store_.pair(mid), lane_store_.pair(mid) + Arena::footprint(words)};
        std::uint32_t* const sums[2] = {out_length >= 2 * length ? out + words : out, stack.take(words)};  // the sum pass is pointwise
        const std::size_t sides[2][2] = {{lo, mid}, {mid, hi}};
        for (int side = 0; side < 2; ++side) {
            const std::size_t xlo = sides[side][0], xhi = sides[side][1], degree = tree.degree(xlo, xhi);
            const std::size_t inner = 8 * LaneLayout::length(std::uint32_t(degree)), inner_mark = stack.mark();
            std::uint32_t* const p = stack.take(words);
            TreeTransform::pointwise_products(s, stored[1 - side], {p, words});
            if (xhi - xlo == 1 || degree <= LaneLayout::kBase) {
                coefficients(p, words, u);
                const std::uint32_t* const sum = base.run(xlo, xhi, p + words - 8 * degree);
                t_.forward({sum, 8 * degree}, 0, {sums[side], words});
            } else if (2 * inner == words) {
                halve(p, words);
                lanes(xlo, xhi, p, half(u), stack, base, sums[side], length);
            } else {
                coefficients(p, words, u);
                std::uint32_t* const q = stack.take(inner);
                t_.forward({p + words - inner, inner}, 0, {q, inner});
                lanes(xlo, xhi, q, 1, stack, base, sums[side], length);
            }
            stack.release(inner_mark);
        }
        TreeTransform::pointwise_product_sums(sums[0], stored[1], sums[1], stored[0], {out, words});
        double_to(out, words, out_length / length, stack);
        stack.release(mark);
    }

    // Node [lo, hi) of the top tree: the transform of its sum of length out_length >= its length
    // into out, from the leaves' sums (count_ coefficients each, leaf l at columns + l stride).
    void top(std::size_t lo, std::size_t hi, const std::uint32_t* columns, std::size_t stride, detail::Stack& stack,
             std::uint32_t* out, std::size_t out_length) {
        const TopTree& tree = *top_;
        if (hi - lo == 1) return t_.forward({columns + lo * stride, count_}, 0, {out, out_length});
        const std::size_t length = StandardLayout::length(tree.degree(lo, hi)), mid = tree.split(lo, hi), mark = stack.mark();
        const std::uint32_t* const stored[2] = {top_store_.pair(mid), top_store_.pair(mid) + Arena::footprint(length)};
        std::uint32_t* const sums[2] = {out_length >= 2 * length ? out + length : out, stack.take(length)};  // the sum pass is leafwise
        top(lo, mid, columns, stride, stack, sums[0], length);
        top(mid, hi, columns, stride, stack, sums[1], length);
        t_.leaf_product_sums(sums[0], stored[1], sums[1], stored[0], {out, length});
        double_to(out, length, out_length / length, stack);
        stack.release(mark);
    }

    // out[0, words) is the transform of a polynomial that it holds without wrap; out[words, factor
    // words) becomes the rest of its transform of factor times the length (forward_upper).
    void double_to(std::uint32_t* out, std::size_t words, std::size_t factor, detail::Stack& stack) const {
        if (factor == 1) return;
        std::uint32_t* const c = stack.take(words);
        t_.inverse({out, words}, {c, words});
        for (std::size_t n = words; n < factor * words; n *= 2) t_.forward_upper({c, words}, 0, {out + n, n});
    }

    std::size_t m_;
};

// Arena words interpolate() takes for m points.
inline std::size_t interpolate_words(std::size_t m) {
    const std::size_t m8 = (m + 7) / 8 * 8;
    const detail::Division sizes(m, m8);
    return PointTree::words(m) + Transform::words(sizes.log()) + Arena::footprint(std::max(m + 1, m8)) +
           Arena::footprint(StandardLayout::length(std::uint32_t(m8))) +
           Arena::footprint(std::max(sizes.scratch_words(), InterpolationTree::scratch_words(m)));
}

// c = the coefficients of the f of degree < m with f(points[i]) = values[i], for m = points.size()
// = values.size() = c.size() >= 1 distinct points; all canonical. arena: interpolate_words(m) words.
inline void interpolate(Arena& arena, std::span<const std::uint32_t> points, std::span<const std::uint32_t> values,
                        std::span<std::uint32_t> c) {
    const std::size_t m = points.size();
    InterpolationTree tree(arena, points);
    const std::size_t m8 = tree.size(), length = StandardLayout::length(std::uint32_t(m8));
    // y: first M' in Montgomery form (M[j] = Q[m - j], M'[j] = (j + 1) M[j + 1]), then the values.
    const std::span<std::uint32_t> y = arena.take(std::max(m + 1, m8));
    std::reverse_copy(tree.product().begin(), tree.product().begin() + std::ptrdiff_t(m + 1), y.begin());
    derivative(y.first(m + 1), y.first(m));
    const detail::Division sizes(m, m8);
    const Transform t(arena, sizes.log());
    const std::span<std::uint32_t> window = arena.take(length);  // the root's window, then R
    const std::span<std::uint32_t> scratch = arena.take(std::max(sizes.scratch_words(), InterpolationTree::scratch_words(m)));
    detail::divide(t, sizes, y.first(m), tree.product(), tree.product_transform(), window.first(m8), scratch);  // rev(M') / Q
    std::copy(values.begin(), values.end(), y.begin());
    y[m] = 0;  // the rest is zero
    tree.interpolate(window.first(m8), y.first(m8), window, scratch);
    std::reverse_copy(window.begin(), window.begin() + std::ptrdiff_t(m), c.begin());
}

}  // namespace poly
