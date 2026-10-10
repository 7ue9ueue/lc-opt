// Tangent Graeffe transforms and zeros on a subgroup, modulo P = 998244353 = 119 2^23 + 1, for
// polynomial_root_finding. x86-64 with AVX2. Design: notes.md.
//
//   roots::Graeffe graeffe(t, arena, d_max);  // t: lg_max >= log2(2 length(d_max))
//   graeffe.run(a, b, k);                     // a: d + 1 coefficients, b: d, d <= d_max
//       // A + eps B = (a + eps b)(x) (a + eps b)(-x) as a polynomial in x^2, k times: a = A and
//       // b = B / 2^k, both times one common factor. With b = a', the roots r of a become r^N,
//       // N = 2^k, and at a simple root z = r^N of A: r = z A'(z) / (B(z) / N).
//   roots::Subgroup h(arena, d_max);
//   h.zeros(l, a, x, b, sink);                // sink(x(z), b(z)) for each zero z of a in the
//                                             // subgroup of order 119 2^l; values times one
//                                             // common factor. d + 1 <= 16 2^l coefficients.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/product_tree.hpp"
#include "lib/poly/transform.hpp"

namespace roots {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

namespace detail {

using poly::detail::add;
using poly::detail::broadcast;
using poly::detail::canonical;
using poly::detail::Factor;
using poly::detail::Factors;
using poly::detail::kP;
using poly::detail::kR;
using poly::detail::load;
using poly::detail::montgomery;
using poly::detail::reduce;
using poly::detail::store;
using poly::detail::times;
using poly::detail::transpose8;
using poly::detail::Vec;

// A vector and its odd lanes moved to the even ones: the operands of _mm256_mul_epu32.
struct Operand {
    Vec even, odd;
};

inline Operand operand(Vec x) { return {x, _mm256_srli_epi64(x, 32)}; }

// 64-bit sums of products, the even lanes' and the odd lanes'.
struct Sum {
    Vec even, odd;
};

inline Sum operator*(const Operand& x, const Operand& y) {
    return {_mm256_mul_epu32(x.even, y.even), _mm256_mul_epu32(x.odd, y.odd)};
}

inline Sum operator+(const Sum& s, const Sum& t) {
    return {_mm256_add_epi64(s.even, t.even), _mm256_add_epi64(s.odd, t.odd)};
}

// s / 2^32 mod P, canonical, for sums below 8 P^2 (the Montgomery term keeps them below 2^64).
inline Vec reduce(const Sum& s) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
    const Vec even = _mm256_add_epi64(s.even, _mm256_mul_epu32(_mm256_mul_epu32(s.even, ni), p));
    const Vec odd = _mm256_add_epi64(s.odd, _mm256_mul_epu32(_mm256_mul_epu32(s.odd, ni), p));
    return canonical(_mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA));  // < 3P
}

// s / 2^32 mod P, canonical, for any sums: 2^32 h + l -> h (2^32 mod P) + l first.
inline Vec reduce_long(const Sum& s) {
    const Vec r = broadcast(kR), zero = _mm256_setzero_si256();
    const auto fold = [&](Vec x) {
        return _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 32), r), _mm256_blend_epi32(x, zero, 0xAA));
    };
    return reduce(Sum{fold(s.even), fold(s.odd)});
}

// Leaves l of a and b mod (x^8 - w_l), lane l of a[j] and b[j] holding coefficient j. With
// a = E(y) + x O(y), b = F(y) + x G(y), y = x^2, the even parts mod (y^4 - w):
//   E^2 - y O^2 = u + w v,  E F - y O G = s + w t   (4 coefficients each).
// out = u, v, s, t, canonical, times 2^-32.
inline void even_parts(const Vec (&a)[8], const Vec (&b)[8], Vec (&out)[16]) {
    const Vec p = broadcast(kP);
    const auto twice = [](Vec x) { return operand(_mm256_add_epi32(x, x)); };  // < 2P
    const auto minus = [p](Vec x) { return operand(_mm256_sub_epi32(p, x)); };  // in (0, P]
    const Operand e0 = operand(a[0]), e1 = operand(a[2]), e2 = operand(a[4]), e3 = operand(a[6]);
    const Operand o0 = operand(a[1]), o1 = operand(a[3]), o2 = operand(a[5]), o3 = operand(a[7]);
    const Operand n0 = minus(a[1]), n1 = minus(a[3]), n2 = minus(a[5]), n3 = minus(a[7]);
    const Operand d1 = twice(a[2]), d2 = twice(a[4]), d3 = twice(a[6]);
    const Operand q1 = twice(a[3]), q2 = twice(a[5]), q3 = twice(a[7]);
    // E^2 - y O^2: products below 2 P^2, at most 4 per sum.
    out[0] = reduce(e0 * e0);
    out[1] = reduce(e0 * d1 + n0 * o0);
    out[2] = reduce(e0 * d2 + e1 * e1 + n0 * q1);
    out[3] = reduce(e0 * d3 + e1 * d2 + n0 * q2 + n1 * o1);
    out[4] = reduce(e1 * d3 + e2 * e2 + n0 * q3 + n1 * q2);
    out[5] = reduce(e2 * d3 + n1 * q3 + n2 * o2);
    out[6] = reduce(e3 * e3 + n2 * q3);
    out[7] = reduce(n3 * o3);
    // E F - y O G: products below P^2, at most 7 per sum.
    const Operand f0 = operand(b[0]), f1 = operand(b[2]), f2 = operand(b[4]), f3 = operand(b[6]);
    const Operand g0 = operand(b[1]), g1 = operand(b[3]), g2 = operand(b[5]), g3 = operand(b[7]);
    out[8] = reduce(e0 * f0);
    out[9] = reduce(e0 * f1 + e1 * f0 + n0 * g0);
    out[10] = reduce(e0 * f2 + e1 * f1 + e2 * f0 + n0 * g1 + n1 * g0);
    out[11] = reduce(e0 * f3 + e1 * f2 + e2 * f1 + e3 * f0 + n0 * g2 + n1 * g1 + n2 * g0);
    out[12] = reduce(e1 * f3 + e2 * f2 + e3 * f1 + n0 * g3 + n1 * g2 + n2 * g1 + n3 * g0);
    out[13] = reduce(e2 * f3 + e3 * f2 + n1 * g3 + n2 * g2 + n3 * g1);
    out[14] = reduce(e3 * f3 + n2 * g3 + n3 * g2);
    out[15] = reduce(n3 * g3);
}

// One Graeffe step on transforms of length 2L (a, b: 2L words): leaves 2q and 2q + 1 (weights
// r = r[q] and -r) give leaf q (weight r^2) of the transform of length L of the result, written
// over a[0, L) and b[0, L). With C = u + w v on leaf 2q and C' = u' - w v' on leaf 2q + 1, the
// result mod (y^8 - r^2) is (C + C') + y^4 (C - C') / r, without the factor 1/2.
inline void pair_step(const poly::Transform& t, u32* a, u32* b, std::size_t length) {
    const Vec p = broadcast(kP);
    for (std::size_t g = 0; g < length / 64; ++g) {
        Vec x[2][8], y[2][8];  // [even leaves, odd leaves][coefficient], lane: pair 8g + lane
        for (int i = 0; i < 8; ++i)
            for (int s = 0; s < 2; ++s) {
                x[s][i] = load(a + 128 * g + 16 * i + 8 * s);
                y[s][i] = load(b + 128 * g + 16 * i + 8 * s);
            }
        for (int s = 0; s < 2; ++s) transpose8(x[s]), transpose8(y[s]);
        Vec c[2][16];
        even_parts(x[0], y[0], c[0]);
        even_parts(x[1], y[1], c[1]);
        const Factors r = poly::detail::entries(t.roots(), 8 * g), r_inv = poly::detail::entries(t.inverse_roots(), 8 * g);
        Vec out[2][8];
        for (int f = 0; f < 2; ++f)
            for (int k = 0; k < 4; ++k) {
                const Vec u0 = c[0][8 * f + k], v0 = c[0][8 * f + 4 + k], u1 = c[1][8 * f + k], v1 = c[1][8 * f + 4 + k];
                const Vec lo = add(add(u0, u1), times(add(_mm256_sub_epi32(v0, v1), p), r));
                const Vec hi = add(times(add(_mm256_sub_epi32(u0, u1), p), r_inv), add(v0, v1));
                out[f][k] = canonical(lo), out[f][4 + k] = canonical(hi);
            }
        transpose8(out[0]), transpose8(out[1]);
        for (int i = 0; i < 8; ++i) store(a + 64 * g + 8 * i, out[0][i]), store(b + 64 * g + 8 * i, out[1][i]);
    }
}

}  // namespace detail

// Transform length for degree d.
inline std::size_t graeffe_length(std::size_t d) { return std::max<std::size_t>(64, std::bit_ceil(d + 1)); }

class Graeffe {
public:
    Graeffe(const poly::Transform& t, poly::Arena& arena, std::size_t d_max) : t_(t) {
        const std::size_t length = graeffe_length(d_max);
        ta_ = arena.take(2 * length).data(), tb_ = arena.take(2 * length).data();
        ca_ = arena.take(length).data(), cb_ = arena.take(length).data();
    }

    void run(std::span<u32> a, std::span<u32> b, int k) const {
        const std::size_t d = b.size(), n = graeffe_length(d);
        const std::span<u32> ta(ta_, 2 * n), tb(tb_, 2 * n), ca(ca_, n), cb(cb_, n);
        t_.forward(a, 0, ta.first(n));
        t_.forward_upper(a, 0, ta.subspan(n));
        t_.forward(b, 0, tb.first(n));
        t_.forward_upper(b, 0, tb.subspan(n));
        for (int step = 1; step <= k; ++step) {
            detail::pair_step(t_, ta_, tb_, n);
            t_.inverse(ta.first(n), ca);
            t_.inverse(tb.first(n), cb);
            if (step == k) break;
            t_.forward_upper(ca.first(d + 1), 0, ta.subspan(n));
            t_.forward_upper(cb.first(d), 0, tb.subspan(n));
        }
        std::copy_n(ca_, d + 1, a.data());
        std::copy_n(cb_, d, b.data());
    }

private:
    const poly::Transform& t_;
    u32 *ta_, *tb_, *ca_, *cb_;
};

// The subgroup H of order 119 2^l as 119 cosets v W of W, the 2^l-th roots of unity, v = u^c for
// u of order 119; cosets 8g .. 8g + 7 in the lanes of group g. On a coset, with n = 2^l and
// V = v^n, a(v w) is the transform of length n of c_j = v^j sum_t a[j + n t] V^t.
class Subgroup {
public:
    static constexpr int kMaxLog = 12;

    Subgroup(poly::Arena& arena, std::size_t d_max) {
        using detail::kP;
        const int l_max = std::min(kMaxLog, std::max(0, int(std::bit_width(d_max)) - 4));
        const std::size_t n = std::size_t(1) << l_max;
        for (auto& c : padded_) c = arena.take(d_max + 1 + 2 * n + 8).data();  // terms * n <= d + n
        for (auto& v : values_) v = arena.take(8 * n).data();
        twiddles_ = arena.take(2 * n).data(), quotients_ = arena.take(2 * n).data();
        // twiddles_[h + j] = w_2h^j for j < h: the butterflies of half-length h.
        for (std::size_t h = 1; h < n; h *= 2) {
            const u32 w = ntt::detail::power(3, (kP - 1) / u32(2 * h));
            u32 x = 1;
            for (std::size_t j = 0; j < h; ++j, x = ntt::detail::multiply_mod(x, w))
                twiddles_[h + j] = x, quotients_[h + j] = ntt::detail::quotient(x);
        }
        // u^c 2^32 for c < 120 (c = 119 pads the last group).
        const u32 u = ntt::detail::power(3, (kP - 1) / 119);
        u32 x = detail::kR;
        for (int c = 0; c < 120; ++c, x = ntt::detail::multiply_mod(x, u)) coset_[c] = x;
    }

    // d + 1 = a.size() = x.size() = b.size() + 1 <= 16 2^l coefficients, l <= the l_max of d_max.
    template <class Sink>
    void zeros(int l, std::span<const u32> a, std::span<const u32> x, std::span<const u32> b, Sink sink) const {
        using namespace detail;
        const std::size_t n = std::size_t(1) << l, terms = (a.size() + n - 1) / n;
        const std::span<const u32> in[3] = {a, x, b};
        for (int i = 0; i < 3; ++i) {
            std::copy(in[i].begin(), in[i].end(), padded_[i]);
            std::fill(padded_[i] + in[i].size(), padded_[i] + terms * n, 0);
        }
        for (int group = 0; group < 15; ++group) {
            twist(group, n, terms);
            for (u32* v : values_) transform(v, n);
            scan(group, n, sink);
        }
    }

private:
    // values_[i][j] = c_j for polynomial i on the cosets of group, times 2^-32.
    void twist(int group, std::size_t n, std::size_t terms) const {
        using namespace detail;
        const Vec v = load(coset_ + 8 * group);  // v 2^32
        Vec big = v;                              // V 2^32 = v^n 2^32
        for (std::size_t m = 1; m < n; m *= 2) big = reduce(montgomery(big, big), kP);
        Operand power[16];  // V^t 2^32
        Vec y = broadcast(kR);
        for (std::size_t t = 0; t < terms; ++t, y = reduce(montgomery(y, big), kP)) power[t] = operand(y);
        Vec f = broadcast(kR);  // v^j 2^32
        for (std::size_t j = 0; j < n; ++j) {
            Sum s[3];
            for (int i = 0; i < 3; ++i) {
                const Vec c = broadcast(padded_[i][j]);
                s[i] = Operand{c, c} * power[0];
            }
            for (std::size_t t = 1; t < terms; ++t)
                for (int i = 0; i < 3; ++i) {
                    const Vec c = broadcast(padded_[i][j + n * t]);
                    s[i] = s[i] + Operand{c, c} * power[t];
                }
            for (int i = 0; i < 3; ++i) store(values_[i] + 8 * j, montgomery(reduce_long(s[i]), f));  // < 2P
            f = reduce(montgomery(f, v), kP);
        }
    }

    // In place, vectors of n lanes: the transform of length n (decimation in frequency, outputs
    // in bit-reversed order), values < 2P.
    void transform(u32* c, std::size_t n) const {
        using namespace detail;
        const Vec p2 = broadcast(2 * kP);
        for (std::size_t h = n / 2; h >= 1; h /= 2)
            for (std::size_t start = 0; start < n; start += 2 * h)
                for (std::size_t j = 0; j < h; ++j) {
                    u32 *x = c + 8 * (start + j), *y = x + 8 * h;
                    const Vec a = load(x), b = load(y);
                    store(x, reduce(add(a, b), 2 * kP));
                    store(y, times(_mm256_sub_epi32(add(a, p2), b), Factor(twiddles_[h + j], quotients_[h + j])));
                }
    }

    template <class Sink>
    void scan(int group, std::size_t n, Sink& sink) const {
        using namespace detail;
        const unsigned valid = group == 14 ? 0x7F : 0xFF;
        for (std::size_t j = 0; j < n; ++j) {
            const Vec a = reduce(load(values_[0] + 8 * j), kP);
            unsigned mask = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(a, _mm256_setzero_si256())))) & valid;
            if (!mask) continue;
            alignas(32) u32 x[8], b[8];
            store(x, reduce(load(values_[1] + 8 * j), kP));
            store(b, reduce(load(values_[2] + 8 * j), kP));
            for (; mask; mask &= mask - 1) {
                const int lane = std::countr_zero(mask);
                sink(x[lane], b[lane]);
            }
        }
    }

    u32* padded_[3];
    u32* values_[3];
    u32 *twiddles_, *quotients_;
    alignas(32) u32 coset_[120];
};

}  // namespace roots
