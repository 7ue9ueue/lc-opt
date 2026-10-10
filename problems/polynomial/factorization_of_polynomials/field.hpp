// Arithmetic modulo an odd prime p < 2^30 in Montgomery form (x stored as x 2^32 mod p), and dense
// kernels on short coefficient arrays: linear combinations of columns, products, a x + b y.
//
// Lazy sums: a u64 accumulator after fold() is at most (2^32 - 1) p and takes `chunk` more
// products of two residues (each at most (p - 1)^2) before it must be folded again; reduce()
// takes any folded value. Vector kernels hold 8 residues per register and work on the even and
// odd 32-bit lanes separately (vpmuludq reads the low half of each 64-bit lane).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace factor {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

class Field {
public:
    explicit Field(u32 modulus) : p(modulus) {
        u32 inverse = p;  // p^-1 mod 2^32 by Newton; p p = 1 mod 8 gives 3 bits
        for (int i = 0; i < 4; ++i) inverse *= 2 - p * inverse;
        neg_inv = -inverse;
        one = u32((u64(1) << 32) % p);
        r2 = u32(u64(one) * one % p);
        const u64 folded = u64(0xFFFFFFFF) * p, square = u64(p - 1) * (p - 1);
        chunk = int(std::min<u64>((~u64(0) - folded) / square, 1 << 20));
        vp = _mm256_set1_epi64x(p);
        vneg_inv = _mm256_set1_epi64x(neg_inv);
        vone = _mm256_set1_epi64x(one);
    }

    u32 p;
    u32 neg_inv;  // -p^-1 mod 2^32
    u32 one;      // 2^32 mod p: 1 in Montgomery form, and the fold factor
    u32 r2;       // 2^64 mod p
    int chunk;    // products a folded accumulator takes
    __m256i vp, vneg_inv, vone;

    static u64 fold(u64 x, u32 one) { return (x >> 32) * one + u32(x); }
    u64 fold(u64 x) const { return fold(x, one); }

    // x / 2^32 mod p, for x <= (2^32 - 1) p.
    u32 reduce(u64 x) const {
        const u32 m = u32(x) * neg_inv;
        const u32 t = u32((x + u64(m) * p) >> 32);
        return t >= p ? t - p : t;
    }
    u32 reduce_lazy(u64 x) const { return reduce(fold(x)); }

    u32 mul(u32 a, u32 b) const { return reduce(u64(a) * b); }
    u32 add(u32 a, u32 b) const { return a + b >= p ? a + b - p : a + b; }
    u32 sub(u32 a, u32 b) const { return a >= b ? a - b : a + p - b; }
    u32 neg(u32 a) const { return a ? p - a : 0; }
    u32 to(u32 a) const { return mul(a, r2); }  // a < p
    u32 from(u32 a) const { return reduce(a); }
    u32 power(u32 a, u64 e) const {
        u32 r = one;
        for (; e; e >>= 1, a = mul(a, a))
            if (e & 1) r = mul(r, a);
        return r;
    }
    u32 inverse(u32 a) const { return power(a, p - 2); }

    // Vector forms of fold and reduce on 64-bit lanes.
    __m256i fold(__m256i x) const {
        const __m256i high = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), vone);
        return _mm256_add_epi64(high, _mm256_and_si256(x, _mm256_set1_epi64x(0xFFFFFFFF)));
    }
    __m256i reduce(__m256i x) const {
        const __m256i m = _mm256_mul_epu32(x, vneg_inv);
        const __m256i t = _mm256_srli_epi64(_mm256_add_epi64(x, _mm256_mul_epu32(m, vp)), 32);
        return _mm256_min_epu32(t, _mm256_sub_epi32(t, vp));
    }
    // Reduced even and odd lanes back into 8 residues.
    static __m256i pack(__m256i even, __m256i odd) { return _mm256_or_si256(even, _mm256_slli_epi64(odd, 32)); }
};

inline int round8(int n) { return (n + 7) & ~7; }

inline __m256i load(const u32* a) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)); }
inline void store(u32* a, __m256i x) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(a), x); }

// out[t] = a x[t] + b y[t] for t < 8 ceil(len / 8): Montgomery residues in and out; out may
// alias x or y.
inline void lincomb(const Field& F, u32* out, const u32* x, u32 a, const u32* y, u32 b, int len) {
    const __m256i va = _mm256_set1_epi32(int(a)), vb = _mm256_set1_epi32(int(b));
    for (int t = 0; t < len; t += 8) {
        const __m256i u = load(x + t), v = load(y + t);
        const __m256i even = _mm256_add_epi64(_mm256_mul_epu32(u, va), _mm256_mul_epu32(v, vb));
        const __m256i odd = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(u, 32), va),
                                             _mm256_mul_epu32(_mm256_srli_epi64(v, 32), vb));
        store(out + t, Field::pack(F.reduce(even), F.reduce(odd)));  // sums <= 2 (p - 1)^2
    }
}

namespace detail {

alignas(32) inline constexpr u32 kZeros[32] = {};

// out[t] = scale init[t] + sum_s coef[s] column_s[t], t < 8 V, with column_s = base + s stride.
template <int V>
inline void combine_block(const Field& F, u32* out, const u32* init, u32 scale, const u32* coef, int count,
                          const u32* base, std::ptrdiff_t stride) {
    __m256i even[V], odd[V];
    const __m256i vscale = _mm256_set1_epi32(int(scale));
#pragma GCC unroll 4
    for (int v = 0; v < V; ++v) {
        const __m256i x = load(init + 8 * v);
        even[v] = _mm256_mul_epu32(x, vscale);
        odd[v] = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), vscale);
    }
    const int chunk = F.chunk;
    for (int s0 = 0; s0 < count; s0 += chunk) {
        const int s1 = std::min(count, s0 + chunk);
        for (int s = s0; s < s1; ++s) {
            const __m256i c = _mm256_set1_epi32(int(coef[s]));
            const u32* column = base + s * stride;
#pragma GCC unroll 4
            for (int v = 0; v < V; ++v) {
                const __m256i x = load(column + 8 * v);
                even[v] = _mm256_add_epi64(even[v], _mm256_mul_epu32(c, x));
                odd[v] = _mm256_add_epi64(odd[v], _mm256_mul_epu32(c, _mm256_srli_epi64(x, 32)));
            }
        }
        if (s1 == count) break;
#pragma GCC unroll 4
        for (int v = 0; v < V; ++v) even[v] = F.fold(even[v]), odd[v] = F.fold(odd[v]);
    }
#pragma GCC unroll 4
    for (int v = 0; v < V; ++v) store(out + 8 * v, Field::pack(F.reduce(F.fold(even[v])), F.reduce(F.fold(odd[v]))));
}

}  // namespace detail

// out[t] = scale init[t] + sum_{s < count} coef[s] (base + s stride)[t] for t < 8 ceil(len / 8);
// init may be null (zero). Columns and init are read over the same rounded range; out may be init
// or a column.
inline void combine_scaled(const Field& F, u32* out, int len, const u32* init, u32 scale, const u32* coef, int count,
                           const u32* base, std::ptrdiff_t stride) {
    const int vectors = (len + 7) / 8;
    int v = 0;
    for (; v + 4 <= vectors; v += 4)
        detail::combine_block<4>(F, out + 8 * v, init ? init + 8 * v : detail::kZeros, scale, coef, count, base + 8 * v,
                                 stride);
    const u32* rest = init ? init + 8 * v : detail::kZeros;
    switch (vectors - v) {
        case 3: detail::combine_block<3>(F, out + 8 * v, rest, scale, coef, count, base + 8 * v, stride); break;
        case 2: detail::combine_block<2>(F, out + 8 * v, rest, scale, coef, count, base + 8 * v, stride); break;
        case 1: detail::combine_block<1>(F, out + 8 * v, rest, scale, coef, count, base + 8 * v, stride); break;
        default: break;
    }
}

// out[t] = init[t] + sum_{s < count} coef[s] (base + s stride)[t], as combine_scaled.
inline void combine(const Field& F, u32* out, int len, const u32* init, const u32* coef, int count, const u32* base,
                    std::ptrdiff_t stride) {
    combine_scaled(F, out, len, init, F.one, coef, count, base, stride);
}

inline constexpr int kProductCap = 256;  // coefficients of a product buffer (degree <= 2 * 100)

// out[0, na + nb - 1) = a b, zero up to the next multiple of 8 (32 when na + nb > 25); na, nb in
// [1, 128]. Blocks of 32 outputs, or one block of 8, 16 or 24.
inline void multiply(const Field& F, const u32* a, int na, const u32* b, int nb, u32* out) {
    alignas(32) u32 padded[32 + 128 + 32];  // b at offset 32, zeros around it
    std::fill(padded, padded + 32, 0);
    std::copy(b, b + nb, padded + 32);
    std::fill(padded + 32 + nb, padded + 64 + nb, 0);
    const int len = na + nb - 1;
    // Column s = b shifted by lo + s: (padded + 32 + t0 - lo - s)[t] = b[t0 + t - lo - s].
    switch ((len + 7) / 8) {
        case 1: return detail::combine_block<1>(F, out, detail::kZeros, F.one, a, na, padded + 32, -1);
        case 2: return detail::combine_block<2>(F, out, detail::kZeros, F.one, a, na, padded + 32, -1);
        case 3: return detail::combine_block<3>(F, out, detail::kZeros, F.one, a, na, padded + 32, -1);
        default: break;
    }
    for (int t0 = 0; t0 < len; t0 += 32) {
        const int lo = std::max(0, t0 - nb + 1), hi = std::min(na, t0 + 32);
        detail::combine_block<4>(F, out + t0, detail::kZeros, F.one, a + lo, hi - lo, padded + 32 + t0 - lo, -1);
    }
}

}  // namespace factor
