// Tangent Graeffe transforms and zeros on cosets of roots of unity, modulo P = 998244353 =
// 119 2^23 + 1, for polynomial_root_finding. x86-64 with AVX2. Design: notes.md.
//
//   roots::Graeffe graeffe(t, arena, d_max);  // t: lg_max >= log2(2 graeffe_length(d_max))
//   graeffe.run(a, b, k, saves);              // a: d + 1 coefficients, b: d, d <= d_max
//       // A + eps B = (a + eps b)(x) (a + eps b)(-x) as a polynomial in x^2, k times: a = A and
//       // b = B / 2^k, both times one common factor; saves get the iterates of their levels.
//       // With b = a', the roots r of a become r^N, N = 2^k, and at a simple root z = r^N of
//       // A: r = z A'(z) / (B(z) / N).
//   roots::CosetZeros cosets(t, arena, d_max, log_max);  // t: lg_max >= log_max + 3
//   cosets.prepare(polys, count, size, log_n);  // up to 3 polynomials of size <= d_max + 1
//   cosets.evaluate(which, v);                  // lane l: the coset v_l W, W the n-th roots of
//       // unity (n = 2^log_n, v in Montgomery form); value(i, q), lane l: polynomial i (bit i
//       // of which) at v_l w^bitrev(q).
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

// Where run() copies the iterate of a level (d + 1 and d words).
struct Save {
    int level;
    u32 *a, *b;
};

class Graeffe {
public:
    Graeffe(const poly::Transform& t, poly::Arena& arena, std::size_t d_max) : t_(t) {
        const std::size_t length = graeffe_length(d_max);
        ta_ = arena.take(2 * length).data(), tb_ = arena.take(2 * length).data();
        ca_ = arena.take(length).data(), cb_ = arena.take(length).data();
    }

    void run(std::span<u32> a, std::span<u32> b, int k, std::span<const Save> saves) const {
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
            for (const Save& s : saves)
                if (s.level == step) std::copy_n(ca_, d + 1, s.a), std::copy_n(cb_, d, s.b);
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

// On a coset v W of the n-th roots of unity, n = 2^log_n, with V = v^n, a(v w) is the transform
// of length n of c_j = v^j sum_t a[j + n t] V^t. Eight cosets in the lanes of a vector; the
// transform of the lanes layout (word 8 j + lane: c_j of that lane) is Transform's of length 8n,
// whose output q is the value at v w^bitrev(q), w = 3^((p - 1) / n).
class CosetZeros {
public:
    static constexpr int kPolys = 3;

    CosetZeros(const poly::Transform& t, poly::Arena& arena, std::size_t d_max, int log_max) : t_(t) {
        const std::size_t n = std::size_t(1) << log_max;
        for (auto& r : rows_) r = arena.take(d_max + 1 + n).data();  // terms * n <= d + n
        for (auto& v : values_) v = arena.take(std::max<std::size_t>(64, 8 * n)).data();
        powers_ = arena.take(16 * (d_max + 2)).data();  // V^t, both operands, t <= d_max + 1
        twiddles_ = arena.take(n).data(), quotients_ = arena.take(n).data();
        // twiddles_[h + j] = w_2h^j for j < h: the butterflies of half-length h.
        for (std::size_t h = 1; h < n; h *= 2) {
            const u32 w = ntt::detail::power(3, (detail::kP - 1) / u32(2 * h));
            u32 x = 1;
            for (std::size_t j = 0; j < h; ++j, x = ntt::detail::multiply_mod(x, w))
                twiddles_[h + j] = x, quotients_[h + j] = ntt::detail::quotient(x);
        }
    }

    // polys[0, count), count <= 3, of size coefficients (zero past their spans), on cosets of
    // n = 2^log_n points, log_n <= log_max. Row j of polynomial i: a[j + n t] for t < terms.
    void prepare(const std::span<const u32>* polys, int count, std::size_t size, int log_n) {
        log_n_ = log_n;
        const std::size_t n = std::size_t(1) << log_n, terms = (size + n - 1) / n;
        terms_ = terms;
        for (int i = 0; i < count; ++i) {
            const std::span<const u32> a = polys[i];
            u32* const rows = rows_[i];
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t t = 0, at = j; t < terms; ++t, at += n) rows[j * terms + t] = at < a.size() ? a[at] : 0;
        }
    }

    std::size_t points() const { return std::size_t(1) << log_n_; }

    // Lanes: the values of polynomials 0 (which = 1), 1 and 2 (which = 6) or all three (which =
    // 7) on the cosets v W, v in Montgomery form; value(i, q): canonical, in the order above.
    void evaluate(unsigned which, detail::Vec v) {
        powers(v);
        switch (which) {
            case 1: return evaluate<1>({0}, v);
            case 6: return evaluate<2>({1, 2}, v);
            default: return evaluate<3>({0, 1, 2}, v);
        }
    }

    detail::Vec value(int i, std::size_t q) const { return detail::load(values_[i] + 8 * q); }

    // Bit reversal of q < n.
    std::size_t reverse(std::size_t q) const {
        std::size_t r = 0;
        for (int i = 0; i < log_n_; ++i) r = r << 1 | (q >> i & 1);
        return r;
    }

    // w^bitrev(q), the root of unity of output q.
    u32 root(std::size_t q) const {
        const std::size_t n = points(), beta = reverse(q);
        if (n == 1) return 1;
        return beta < n / 2 ? twiddles_[n / 2 + beta] : detail::kP - twiddles_[beta];
    }

private:
    // powers_ = V^t 2^32, t < terms, each as its even and odd operands.
    void powers(detail::Vec v) {
        using namespace detail;
        Vec big = v;
        for (int m = 0; m < log_n_; ++m) big = reduce(montgomery(big, big), kP);
        Vec y = broadcast(kR);
        for (std::size_t t = 0; t < terms_; ++t, y = reduce(montgomery(y, big), kP)) {
            store(powers_ + 16 * t, y);
            store(powers_ + 16 * t + 8, _mm256_srli_epi64(y, 32));
        }
    }

    template <int K>
    void evaluate(const int (&which)[K], detail::Vec v) {
        using namespace detail;
        const std::size_t n = points(), terms = terms_;
        Vec f = broadcast(kR);  // v^j 2^32
        for (std::size_t j = 0; j < n; ++j) {
            Vec total[K];
            for (int i = 0; i < K; ++i) total[i] = _mm256_setzero_si256();
            for (std::size_t t0 = 0; t0 < terms; t0 += 16) {  // 64-bit sums below 16 P^2
                const std::size_t t1 = std::min(terms, t0 + 16);
                Sum s[K];
                for (int i = 0; i < K; ++i) s[i] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
#pragma GCC unroll 4
                for (std::size_t t = t0; t < t1; ++t) {
                    const Operand power{load(powers_ + 16 * t), load(powers_ + 16 * t + 8)};
                    for (int i = 0; i < K; ++i) {
                        const Vec c = broadcast(rows_[which[i]][j * terms + t]);
                        s[i] = s[i] + Operand{c, c} * power;
                    }
                }
                for (int i = 0; i < K; ++i) total[i] = reduce(add(total[i], reduce_long(s[i])), kP);
            }
            for (int i = 0; i < K; ++i) store(values_[which[i]] + 8 * j, reduce(montgomery(total[i], f), kP));
            f = reduce(montgomery(f, v), kP);
        }
        for (int i = 0; i < K; ++i) transform(values_[which[i]]);
    }

    // In place: the transform of length n of the lanes (outputs in bit-reversed order).
    void transform(u32* c) const {
        using namespace detail;
        const std::size_t n = points();
        if (n >= 8) return t_.forward(std::span<u32>(c, 8 * n));
        const Vec p2 = broadcast(2 * kP);
        for (std::size_t h = n / 2; h >= 1; h /= 2)
            for (std::size_t start = 0; start < n; start += 2 * h)
                for (std::size_t j = 0; j < h; ++j) {
                    u32 *x = c + 8 * (start + j), *y = x + 8 * h;
                    const Vec a = load(x), b = load(y);
                    store(x, reduce(add(a, b), 2 * kP));
                    store(y, times(_mm256_sub_epi32(add(a, p2), b), Factor(twiddles_[h + j], quotients_[h + j])));
                }
        for (std::size_t j = 0; j < n; ++j) store(c + 8 * j, reduce(load(c + 8 * j), kP));
    }

    const poly::Transform& t_;
    u32* rows_[kPolys];
    u32* values_[kPolys];
    u32 *powers_, *twiddles_, *quotients_;
    int log_n_ = 0;
    std::size_t terms_ = 0;
};

}  // namespace roots
